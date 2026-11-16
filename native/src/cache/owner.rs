use super::{PROTOCOL, store::Store, wire::MAX_FRAME};
use anyhow::{Result, ensure};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::{thread, time::Duration};

fn serve(
    mut stream: TcpStream,
    store: &Store,
    auth: &[u8; 32],
    stop: &AtomicBool,
    owner: u64,
) -> Result<()> {
    // Windows accepts inherit the listener's nonblocking mode; frames use blocking I/O.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut token = [0; 32];
    stream.read_exact(&mut token)?;
    ensure!(
        constant_time_eq::constant_time_eq(&token, auth),
        "invalid local capability"
    );
    let mut version = [0; 4];
    stream.read_exact(&mut version)?;
    stream.write_all(&PROTOCOL.to_le_bytes())?;
    ensure!(
        u32::from_le_bytes(version) == PROTOCOL,
        "incompatible artifact cache protocol"
    );
    stream.set_read_timeout(None)?;
    loop {
        let mut size = [0; 4];
        if stream.read_exact(&mut size).is_err() {
            return Ok(());
        }
        let size = u32::from_le_bytes(size) as usize;
        ensure!((5..=MAX_FRAME).contains(&size), "invalid frame size");
        let mut frame = vec![0; size];
        stream.read_exact(&mut frame)?;
        let success;
        let mut output = match store.dispatch(&frame, owner) {
            Ok(output) => {
                success = true;
                let mut response = vec![0];
                response.extend_from_slice(&output);
                response
            }
            Err(error) => {
                success = false;
                store.release(owner);
                let mut response = vec![1];
                response.extend_from_slice(error.to_string().as_bytes());
                response
            }
        };
        ensure!(output.len() <= MAX_FRAME + 1, "reply exceeds frame budget");
        let mut reply = Vec::with_capacity(output.len() + 4);
        reply.extend_from_slice(&(output.len() as u32).to_le_bytes());
        reply.append(&mut output);
        stream.write_all(&reply)?;
        if frame[0] == 6 && success {
            stop.store(true, Ordering::Release);
            return Ok(());
        }
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
    }
}

pub fn run(path: &Path, memory: u64, capacity: u64) -> Result<()> {
    let mut auth = [0; 32];
    std::io::stdin().read_exact(&mut auth)?;
    let store = Arc::new(Store::open(path, memory, capacity)?);
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    listener.set_nonblocking(true)?;
    let stop = Arc::new(AtomicBool::new(false));
    let watch = stop.clone();
    thread::spawn(move || {
        let mut byte = [0];
        let _ = std::io::stdin().read(&mut byte);
        watch.store(true, Ordering::Release);
    });
    println!(
        "{}",
        serde_json::json!({"port":listener.local_addr()?.port(),"identity":store.identity,"protocol":PROTOCOL,"backend":"rust"})
    );
    std::io::stdout().flush()?;
    let next = AtomicU64::new(1);
    let mut clients: Vec<(thread::JoinHandle<()>, TcpStream)> = Vec::new();
    while !stop.load(Ordering::Acquire) {
        clients.retain(|(thread, _)| !thread.is_finished());
        match listener.accept() {
            Ok((stream, _)) => {
                let socket = stream.try_clone()?;
                let cache = store.clone();
                let shutdown = stop.clone();
                let owner = next.fetch_add(1, Ordering::Relaxed);
                let job = thread::spawn(move || {
                    let _ = serve(stream, &cache, &auth, &shutdown, owner);
                    cache.release(owner);
                });
                clients.push((job, socket));
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5))
            }
            Err(error) => return Err(error.into()),
        }
    }
    for (_, stream) in &clients {
        let _ = stream.shutdown(Shutdown::Both);
    }
    for (thread, _) in clients {
        let _ = thread.join();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_nonblocking_stream_waits_for_the_complete_handshake() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = Store::open(&directory.path().join("cache.sqlite3"), 4096, 0)?;
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let mut client = TcpStream::connect(listener.local_addr()?)?;
        let (stream, _) = listener.accept()?;
        // Windows inherits the listener's mode; force it here on every platform.
        stream.set_nonblocking(true)?;
        let auth = [17; 32];
        let stop = AtomicBool::new(false);
        thread::scope(|scope| -> Result<()> {
            let worker = scope.spawn(|| serve(stream, &store, &auth, &stop, 1));
            let result = (|| -> Result<()> {
                client.set_read_timeout(Some(Duration::from_millis(100)))?;
                client.write_all(&auth[..16])?;
                let mut byte = [0];
                ensure!(
                    matches!(client.read(&mut byte), Err(error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut)),
                    "owner must wait for the rest of the capability"
                );
                client.set_read_timeout(Some(Duration::from_secs(10)))?;
                client.write_all(&auth[16..])?;
                client.write_all(&PROTOCOL.to_le_bytes())?;
                let mut version = [0; 4];
                client.read_exact(&mut version)?;
                ensure!(
                    u32::from_le_bytes(version) == PROTOCOL,
                    "invalid protocol reply"
                );
                for _ in 0..2 {
                    let reply = super::super::wire::call(&mut client, 5, b"{}", b"")?;
                    let stats: serde_json::Value = serde_json::from_slice(&reply)?;
                    ensure!(stats["backend"] == "rust", "invalid statistics reply");
                }
                Ok(())
            })();
            drop(client);
            let served = worker.join().expect("owner thread panicked");
            result?;
            served
        })
    }
}

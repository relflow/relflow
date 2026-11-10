use anyhow::{Result, ensure};
use native::{store::Store, wire::MAX_FRAME};
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
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut token = [0; 32];
    stream.read_exact(&mut token)?;
    ensure!(
        constant_time_eq::constant_time_eq(&token, auth),
        "invalid local capability"
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: cache-owner DATABASE HOT_BYTES ENTRY_CAP"
    );
    let mut auth = [0; 32];
    std::io::stdin().read_exact(&mut auth)?;
    let store = Arc::new(Store::open(
        Path::new(&args[1]),
        args[2].parse()?,
        args[3].parse()?,
    )?);
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
        serde_json::json!({"port":listener.local_addr()?.port(),"identity":store.identity})
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

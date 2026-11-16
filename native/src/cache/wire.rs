use anyhow::{Result, ensure};
use std::io::{Read, Write};
use std::net::TcpStream;

pub const MAX_FRAME: usize = 32 * 1024 * 1024;
pub const MAX_ROWS: usize = 4096;

pub fn call(stream: &mut TcpStream, op: u8, meta: &[u8], payload: &[u8]) -> Result<Vec<u8>> {
    let size = 5 + meta.len() + payload.len();
    ensure!(
        size <= MAX_FRAME && meta.len() <= 65536,
        "artifact cache request exceeds frame budget"
    );
    let mut frame = Vec::with_capacity(size + 4);
    frame.extend_from_slice(&(size as u32).to_le_bytes());
    frame.push(op);
    frame.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    frame.extend_from_slice(meta);
    frame.extend_from_slice(payload);
    stream.write_all(&frame)?;
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let size = u32::from_le_bytes(header) as usize;
    ensure!(
        (1..=MAX_FRAME + 1).contains(&size),
        "invalid artifact cache response size"
    );
    let mut response = vec![0; size];
    stream.read_exact(&mut response)?;
    ensure!(
        response[0] == 0,
        "{}",
        String::from_utf8_lossy(&response[1..])
    );
    response.remove(0);
    Ok(response)
}

pub fn ids(data: &[u8], n: usize) -> Result<Vec<i64>> {
    ensure!(data.len() == n * 8, "artifact ID batch has incorrect size");
    Ok(data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|row| i64::from_le_bytes(*row))
        .collect())
}

/// Offset tables retain contiguous fixed-width batches and also admit ragged payloads.
pub fn pack<'a>(rows: impl Iterator<Item = &'a [u8]>) -> Result<Vec<u8>> {
    let rows: Vec<_> = rows.collect();
    let size = (rows.len() + 1) * 4 + rows.iter().map(|v| v.len()).sum::<usize>();
    ensure!(
        size <= MAX_FRAME,
        "artifact batch exceeds 32 MiB; reduce batch size or split large artifacts"
    );
    let mut output = Vec::with_capacity(size);
    let mut offset = 0u32;
    output.extend_from_slice(&offset.to_le_bytes());
    for row in &rows {
        offset += row.len() as u32;
        output.extend_from_slice(&offset.to_le_bytes());
    }
    for row in rows {
        output.extend_from_slice(row);
    }
    Ok(output)
}

pub fn unpack(data: &[u8], n: usize) -> Result<(Vec<&[u8]>, usize)> {
    let header = (n + 1) * 4;
    ensure!(data.len() >= header, "truncated artifact offsets");
    let offsets: Vec<usize> = data[..header]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| u32::from_le_bytes(*v) as usize)
        .collect();
    ensure!(
        offsets[0] == 0 && offsets.windows(2).all(|v| v[0] <= v[1]),
        "invalid artifact offsets"
    );
    let size = header + offsets[n];
    ensure!(size <= data.len(), "truncated artifact payload");
    Ok((
        offsets
            .windows(2)
            .map(|v| &data[header + v[0]..header + v[1]])
            .collect(),
        size,
    ))
}

use crate::wire::{MAX_FRAME, MAX_ROWS, ids, pack, unpack};
use anyhow::{Result, bail, ensure};
use moka::sync::Cache;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Clone, Eq, PartialEq, Hash)]
pub enum Key {
    Alias([u8; 32]),
    Result(i64, [u8; 32]),
}
#[derive(Clone)]
pub enum Hot {
    Alias(i64, i64),
    Payload(Arc<[u8]>),
}
#[derive(Default)]
pub struct Counts {
    identified: AtomicU64,
    alias_hits: AtomicU64,
    requests: AtomicU64,
    hits: AtomicU64,
    disk_reads: AtomicU64,
    compute_claims: AtomicU64,
}

type Claim = (i64, String);
pub struct Store {
    connection: Mutex<Connection>,
    hot: Cache<Key, Hot>,
    memory: u64,
    pub identity: String,
    claims: Mutex<HashMap<Claim, u64>>,
    changed: Condvar,
    counts: Counts,
}

impl Store {
    pub fn open(path: &Path, memory: u64, capacity: u64) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(path)?;
        connection.busy_timeout(Duration::from_secs(60))?;
        connection.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-4096; PRAGMA mmap_size=0;
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS aliases (id INTEGER PRIMARY KEY AUTOINCREMENT, key BLOB UNIQUE NOT NULL, object_id INTEGER);
            CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY AUTOINCREMENT, key BLOB UNIQUE NOT NULL, payload BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS results (id INTEGER NOT NULL, namespace TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY (id,namespace)) WITHOUT ROWID;
            INSERT OR IGNORE INTO metadata VALUES ('format','1');
            INSERT OR IGNORE INTO metadata VALUES ('identity',lower(hex(randomblob(32))));
            INSERT OR IGNORE INTO metadata VALUES ('aliases','0');
            INSERT OR IGNORE INTO metadata VALUES ('objects','0');
            INSERT OR IGNORE INTO metadata VALUES ('results','0');")?;
        let format: String =
            connection.query_row("SELECT value FROM metadata WHERE key='format'", [], |r| {
                r.get(0)
            })?;
        ensure!(
            format == "1",
            "unsupported artifact cache format {format}; use a new store directory"
        );
        let identity =
            connection.query_row("SELECT value FROM metadata WHERE key='identity'", [], |r| {
                r.get(0)
            })?;
        let minimum = if capacity == 0 {
            160
        } else {
            160.max(memory.div_ceil(capacity))
        };
        ensure!(
            minimum <= u32::MAX as u64,
            "entry charge exceeds supported cache weight"
        );
        let hot = Cache::builder()
            .max_capacity(memory)
            .weigher(move |_: &Key, value: &Hot| {
                let payload = match value {
                    Hot::Alias(_, _) => 16,
                    Hot::Payload(bytes) => bytes.len() as u64,
                };
                minimum.max(160 + payload) as u32
            })
            .build();
        Ok(Self {
            connection: Mutex::new(connection),
            hot,
            memory,
            identity,
            claims: Mutex::new(HashMap::new()),
            changed: Condvar::new(),
            counts: Counts::default(),
        })
    }

    pub fn identify(&self, keys: &[[u8; 32]]) -> Result<Vec<u8>> {
        self.counts
            .identified
            .fetch_add(keys.len() as u64, Ordering::Relaxed);
        let mut output = vec![0; keys.len() * 16];
        let mut missing = Vec::new();
        for (index, key) in keys.iter().enumerate() {
            if let Some(Hot::Alias(alias, object)) = self.hot.get(&Key::Alias(*key)) {
                output[index * 8..index * 8 + 8].copy_from_slice(&alias.to_le_bytes());
                output[(keys.len() + index) * 8..(keys.len() + index + 1) * 8]
                    .copy_from_slice(&object.to_le_bytes());
                self.counts.alias_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                missing.push(index);
            }
        }
        if missing.is_empty() {
            return Ok(output);
        }
        let mut connection = self.connection.lock().unwrap();
        let transaction = connection.transaction()?;
        let mut inserted = 0;
        let mut admitted = Vec::new();
        {
            let mut select =
                transaction.prepare_cached("SELECT id,object_id FROM aliases WHERE key=?")?;
            let mut insert =
                transaction.prepare_cached("INSERT OR IGNORE INTO aliases(key) VALUES (?)")?;
            for index in missing {
                let key = keys[index].as_slice();
                let mut row = select
                    .query_row([key], |r| {
                        Ok((r.get::<_, i64>(0)?, r.get::<_, Option<i64>>(1)?))
                    })
                    .optional()?;
                if row.is_none() {
                    inserted += insert.execute([key])?;
                    row = Some(select.query_row([key], |r| Ok((r.get(0)?, r.get(1)?)))?);
                }
                let (alias, object) = row.unwrap();
                output[index * 8..index * 8 + 8].copy_from_slice(&alias.to_le_bytes());
                if let Some(object) = object {
                    output[(keys.len() + index) * 8..(keys.len() + index + 1) * 8]
                        .copy_from_slice(&object.to_le_bytes());
                    admitted.push((keys[index], alias, object));
                }
            }
        }
        if inserted > 0 {
            transaction.execute(
                "UPDATE metadata SET value=CAST(value AS INTEGER)+? WHERE key='aliases'",
                [inserted as i64],
            )?;
        }
        transaction.commit()?;
        for (key, alias, object) in admitted {
            self.hot.insert(Key::Alias(key), Hot::Alias(alias, object));
        }
        self.hot.run_pending_tasks();
        Ok(output)
    }

    pub fn bind(
        &self,
        aliases: &[i64],
        keys: &[[u8; 32]],
        rows: &[&[u8]],
        namespace: &str,
    ) -> Result<Vec<u8>> {
        let mut connection = self.connection.lock().unwrap();
        let transaction = connection.transaction()?;
        let mut canonical = Vec::with_capacity(aliases.len());
        let mut inserted = 0;
        {
            let mut alias =
                transaction.prepare_cached("SELECT key,object_id FROM aliases WHERE id=?")?;
            let mut previous =
                transaction.prepare_cached("SELECT key,payload FROM objects WHERE id=?")?;
            let mut insert = transaction
                .prepare_cached("INSERT OR IGNORE INTO objects(key,payload) VALUES (?,?)")?;
            let mut find = transaction.prepare_cached("SELECT id FROM objects WHERE key=?")?;
            let mut update =
                transaction.prepare_cached("UPDATE aliases SET object_id=? WHERE id=?")?;
            for (index, id) in aliases.iter().enumerate() {
                let payload = rows[index];
                let stored = alias
                    .query_row([id], |r| {
                        Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, Option<i64>>(1)?))
                    })
                    .optional()?;
                let (key, ready) =
                    stored.ok_or_else(|| anyhow::anyhow!("alias {id} does not exist"))?;
                ensure!(
                    key == keys[index],
                    "prepared artifact does not match alias {id}"
                );
                let mut hash = Sha256::new();
                hash.update(namespace.as_bytes());
                hash.update([0]);
                hash.update(payload);
                let key: [u8; 32] = hash.finalize().into();
                if let Some(ready) = ready {
                    let (existing_key, existing): (Vec<u8>, Vec<u8>) =
                        previous.query_row([ready], |r| Ok((r.get(0)?, r.get(1)?)))?;
                    ensure!(
                        existing_key == key && existing == payload,
                        "conflicting prepared artifact for alias {id}; use a distinct preprocessing identity"
                    );
                    canonical.push(ready);
                } else {
                    inserted += insert.execute(params![key.as_slice(), payload])?;
                    let ready: i64 = find.query_row([key.as_slice()], |r| r.get(0))?;
                    update.execute(params![ready, id])?;
                    canonical.push(ready);
                }
            }
        }
        if inserted > 0 {
            transaction.execute(
                "UPDATE metadata SET value=CAST(value AS INTEGER)+? WHERE key='objects'",
                [inserted as i64],
            )?;
        }
        transaction.commit()?;
        // Publish only committed aliases; a later conflicting row rolls back the entire batch.
        for ((key, alias), object) in keys.iter().zip(aliases).zip(&canonical) {
            self.hot
                .insert(Key::Alias(*key), Hot::Alias(*alias, *object));
        }
        self.hot.run_pending_tasks();
        Ok(canonical.iter().flat_map(|id| id.to_le_bytes()).collect())
    }

    pub fn fetch(&self, identifiers: &[i64], namespace: &str, owner: u64) -> Result<Vec<u8>> {
        self.counts
            .requests
            .fetch_add(identifiers.len() as u64, Ordering::Relaxed);
        let version: [u8; 32] = Sha256::digest(namespace.as_bytes()).into();
        let mut status = vec![0; identifiers.len()];
        let mut results: Vec<Arc<[u8]>> = Vec::new();
        let mut inputs: Vec<Arc<[u8]>> = Vec::new();
        let mut size = identifiers.len() + 8;
        let connection = self.connection.lock().unwrap();
        let mut result =
            connection.prepare_cached("SELECT payload FROM results WHERE id=? AND namespace=?")?;
        let mut source = connection.prepare_cached("SELECT payload FROM objects WHERE id=?")?;
        let mut claims = self.claims.lock().unwrap();
        let mut owned = claims.values().filter(|id| **id == owner).count();
        for (index, id) in identifiers.iter().enumerate() {
            let key = Key::Result(*id, version);
            let cached = self.hot.get(&key);
            let value = if let Some(Hot::Payload(value)) = cached {
                self.counts.hits.fetch_add(1, Ordering::Relaxed);
                Some(value)
            } else if let Some(value) = result
                .query_row(params![id, namespace], |r| r.get::<_, Vec<u8>>(0))
                .optional()?
            {
                let value: Arc<[u8]> = value.into();
                self.hot.insert(key, Hot::Payload(value.clone()));
                self.counts.disk_reads.fetch_add(1, Ordering::Relaxed);
                Some(value)
            } else {
                None
            };
            if let Some(value) = value {
                size += 4 + value.len();
                ensure!(
                    size <= MAX_FRAME,
                    "artifact lookup exceeds 32 MiB; reduce batch size or split large artifacts"
                );
                status[index] = 1;
                results.push(value);
                continue;
            }
            let claim = (*id, namespace.to_owned());
            if let Some(claimed) = claims.get(&claim) {
                if *claimed != owner {
                    status[index] = 2;
                    continue;
                }
            } else {
                ensure!(
                    owned < MAX_ROWS,
                    "connection exceeds 4096 in-flight computations"
                );
                owned += 1;
            }
            let value = source
                .query_row([id], |r| r.get::<_, Vec<u8>>(0))
                .optional()?
                .ok_or_else(|| anyhow::anyhow!("artifact ID {id} has no prepared input"))?;
            size += 4 + value.len();
            ensure!(
                size <= MAX_FRAME,
                "artifact lookup exceeds 32 MiB; reduce batch size or split large artifacts"
            );
            claims.insert(claim, owner);
            inputs.push(value.into());
            self.counts.compute_claims.fetch_add(1, Ordering::Relaxed);
        }
        drop(claims);
        drop(result);
        drop(source);
        drop(connection);
        let mut output = status;
        output.extend(pack(results.iter().map(|v| v.as_ref()))?);
        output.extend(pack(inputs.iter().map(|v| v.as_ref()))?);
        self.hot.run_pending_tasks();
        Ok(output)
    }

    pub fn publish(
        &self,
        identifiers: &[i64],
        namespace: &str,
        rows: &[&[u8]],
        owner: u64,
    ) -> Result<Vec<u8>> {
        let mut connection = self.connection.lock().unwrap();
        let transaction = connection.transaction()?;
        let mut claims = self.claims.lock().unwrap();
        let mut output = Vec::with_capacity(rows.len());
        let mut inserted = 0;
        {
            let mut insert =
                transaction.prepare_cached("INSERT OR IGNORE INTO results VALUES (?,?,?)")?;
            let mut select = transaction
                .prepare_cached("SELECT payload FROM results WHERE id=? AND namespace=?")?;
            for (index, id) in identifiers.iter().enumerate() {
                ensure!(
                    claims.get(&(*id, namespace.to_owned())) == Some(&owner),
                    "artifact result {id} has no matching compute claim"
                );
                inserted += insert.execute(params![id, namespace, rows[index]])?;
                output.push(select.query_row(params![id, namespace], |r| r.get::<_, Vec<u8>>(0))?);
            }
        }
        let packed = pack(output.iter().map(|v| v.as_slice()))?;
        if inserted > 0 {
            transaction.execute(
                "UPDATE metadata SET value=CAST(value AS INTEGER)+? WHERE key='results'",
                [inserted as i64],
            )?;
        }
        transaction.commit()?;
        for id in identifiers {
            claims.remove(&(*id, namespace.to_owned()));
        }
        self.changed.notify_all();
        Ok(packed)
    }

    pub fn release(&self, owner: u64) {
        self.claims
            .lock()
            .unwrap()
            .retain(|_, value| *value != owner);
        self.changed.notify_all();
    }
    pub fn wait(&self) {
        drop(
            self.changed
                .wait_timeout(self.claims.lock().unwrap(), Duration::from_millis(100))
                .unwrap(),
        );
    }
    pub fn stats(&self) -> Result<Value> {
        let connection = self.connection.lock().unwrap();
        self.hot.run_pending_tasks();
        let count = |name| {
            connection.query_row(
                "SELECT CAST(value AS INTEGER) FROM metadata WHERE key=?",
                [name],
                |r| r.get::<_, i64>(0),
            )
        };
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        Ok(
            json!({"identity":self.identity,"identified":load(&self.counts.identified),"alias_hits":load(&self.counts.alias_hits),"requests":load(&self.counts.requests),"hits":load(&self.counts.hits),"disk_reads":load(&self.counts.disk_reads),"compute_claims":load(&self.counts.compute_claims),"aliases":count("aliases")?,"objects":count("objects")?,"results":count("results")?,"memory":self.memory,"hot_bytes":self.hot.weighted_size(),"hot_entries":self.hot.entry_count(),"page_bytes":4194304,"claims":self.claims.lock().unwrap().len()}),
        )
    }
    pub fn dispatch(&self, frame: &[u8], owner: u64) -> Result<Vec<u8>> {
        ensure!(frame.len() >= 5, "truncated request");
        let op = frame[0];
        let size = u32::from_le_bytes(frame[1..5].try_into()?) as usize;
        ensure!(
            size <= 65536 && frame.len() >= 5 + size,
            "invalid metadata size"
        );
        let meta: Value = serde_json::from_slice(&frame[5..5 + size])?;
        let data = &frame[5 + size..];
        match op {
            5 => return Ok(serde_json::to_vec(&self.stats()?)?),
            6 => return Ok(Vec::new()),
            7 => {
                self.release(owner);
                return Ok(Vec::new());
            }
            8 => {
                self.wait();
                return Ok(Vec::new());
            }
            _ => (),
        }
        let n: usize = meta["n"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("invalid n"))?
            .try_into()?;
        ensure!(n <= MAX_ROWS, "request exceeds 4096 rows");
        if op == 1 {
            ensure!(data.len() == n * 32, "invalid key batch");
            let keys: Vec<[u8; 32]> = data
                .chunks_exact(32)
                .map(|v| v.try_into().unwrap())
                .collect();
            return self.identify(&keys);
        }
        ensure!(data.len() >= n * 8, "truncated ID batch");
        let identifiers = ids(&data[..n * 8], n)?;
        ensure!(
            identifiers.iter().all(|id| *id > 0),
            "artifact IDs must be positive"
        );
        let data = &data[n * 8..];
        let namespace = meta["namespace"]
            .as_str()
            .filter(|v| !v.is_empty() && v.len() <= 1024 && !v.contains('\0'))
            .ok_or_else(|| {
                anyhow::anyhow!("namespace must be nonempty, NUL-free, and at most 1024 bytes")
            })?;
        match op {
            2 => {
                ensure!(data.len() >= n * 32, "truncated alias key batch");
                let keys: Vec<[u8; 32]> = data[..n * 32]
                    .chunks_exact(32)
                    .map(|v| v.try_into().unwrap())
                    .collect();
                let (rows, used) = unpack(&data[n * 32..], n)?;
                ensure!(
                    used == data.len() - n * 32,
                    "trailing prepared artifact bytes"
                );
                self.bind(&identifiers, &keys, &rows, namespace)
            }
            3 => {
                ensure!(data.is_empty(), "unexpected lookup payload");
                self.fetch(&identifiers, namespace, owner)
            }
            4 => {
                let (rows, used) = unpack(data, n)?;
                ensure!(used == data.len(), "trailing result bytes");
                self.publish(&identifiers, namespace, &rows, owner)
            }
            _ => bail!("unknown cache operation {op}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opaque_payloads_deduplicate_and_survive_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("artifacts.sqlite3");
        let keys = [[1; 32], [2; 32]];
        let store = Store::open(&path, 512, 0)?;
        let reply = store.identify(&keys)?;
        let aliases = ids(&reply[..16], 2)?;
        let payloads: &[&[u8]] = &[b"arbitrary variable bytes", b"arbitrary variable bytes"];
        let ready = store.bind(&aliases, &keys, payloads, "prepare/v1")?;
        let objects = ids(&ready, 2)?;
        assert_eq!(objects[0], objects[1]);
        let fresh = store.fetch(&objects[..1], "encoder/v1", 1)?;
        assert_eq!(fresh[0], 0);
        assert_eq!(store.fetch(&objects[..1], "encoder/v1", 2)?[0], 2);
        store.release(1);
        assert_eq!(store.fetch(&objects[..1], "encoder/v1", 2)?[0], 0);
        store.publish(&objects[..1], "encoder/v1", &[b""], 2)?;
        assert_eq!(store.fetch(&objects, "encoder/v1", 1)?[..2], [1, 1]);
        let identity = store.identity.clone();
        drop(store);
        let store = Store::open(&path, 0, 0)?;
        assert_eq!(store.identity, identity);
        assert_eq!(store.fetch(&objects, "encoder/v1", 3)?[..2], [1, 1]);
        assert_eq!(store.stats()?["hot_bytes"], 0);
        Ok(())
    }
    #[test]
    fn failed_binding_is_atomic() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = Store::open(&dir.path().join("artifacts.sqlite3"), 4096, 0)?;
        let keys = [[1; 32], [2; 32]];
        let reply = store.identify(&keys)?;
        let aliases = ids(&reply[..16], 2)?;
        store.bind(&aliases[1..], &keys[1..], &[b"existing"], "v1")?;
        assert!(
            store
                .bind(&aliases, &keys, &[b"new", b"conflict"], "v1")
                .is_err()
        );
        let lookup = store.identify(&keys)?;
        assert_eq!(ids(&lookup[16..], 2)?[0], 0);
        assert_eq!(store.stats()?["objects"], 1);
        Ok(())
    }
}

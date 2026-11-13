use super::wire::{self, MAX_ROWS};
use arrow_array::{
    Array, BinaryArray, BinaryViewArray, FixedSizeBinaryArray, LargeBinaryArray, LargeStringArray,
    StringArray, StringViewArray, make_array,
};
use arrow_data::ArrayData;
use arrow_pyarrow::PyArrowType;
use arrow_schema::DataType;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::Mutex;
use std::time::Duration;

#[pyclass(frozen, module = "relflow_native.native.cache")]
struct Client {
    port: u16,
    auth: [u8; 32],
    stream: Mutex<Option<TcpStream>>,
}
impl Client {
    fn call(&self, op: u8, meta: &[u8], payload: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut stream = self.stream.lock().unwrap();
        if stream.is_none() {
            let mut socket = TcpStream::connect(("127.0.0.1", self.port))?;
            socket.set_read_timeout(Some(Duration::from_secs(300)))?;
            socket.set_write_timeout(Some(Duration::from_secs(300)))?;
            socket.write_all(&self.auth)?;
            socket.write_all(&super::PROTOCOL.to_le_bytes())?;
            let mut version = [0; 4];
            socket.read_exact(&mut version)?;
            anyhow::ensure!(
                u32::from_le_bytes(version) == super::PROTOCOL,
                "incompatible artifact cache protocol; install matching relflow and relflow-native versions"
            );
            *stream = Some(socket);
        }
        wire::call(stream.as_mut().unwrap(), op, meta, payload)
    }
}

#[pymethods]
impl Client {
    #[new]
    fn new(port: u16, auth: &[u8]) -> PyResult<Self> {
        let auth = auth
            .try_into()
            .map_err(|_| PyValueError::new_err("local capability must contain 32 bytes"))?;
        Ok(Self {
            port,
            auth,
            stream: Mutex::new(None),
        })
    }

    fn request<'py>(
        &self,
        py: Python<'py>,
        op: u8,
        meta: &[u8],
        payload: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let response = py
            .detach(|| self.call(op, meta, payload))
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(PyBytes::new(py, &response))
    }

    fn identify_arrow<'py>(
        &self,
        py: Python<'py>,
        input: PyArrowType<ArrayData>,
        namespace: &str,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let array = make_array(input.0);
        if array.null_count() != 0 || array.len() > MAX_ROWS {
            return Err(PyValueError::new_err(
                "identify expects at most 4096 nonnull strings or binary values",
            ));
        }
        let (response, keys) = py
            .detach(|| -> anyhow::Result<_> {
                let mut prefix = Sha256::new();
                prefix.update(namespace.as_bytes());
                prefix.update([0]);
                let mut keys = Vec::with_capacity(array.len() * 32);
                let mut append = |value: &[u8]| {
                    let mut hash = prefix.clone();
                    hash.update(value);
                    keys.extend_from_slice(&hash.finalize());
                };
                match array.data_type() {
                    DataType::Utf8 => {
                        for value in array
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap()
                            .iter()
                            .flatten()
                        {
                            append(value.as_bytes());
                        }
                    }
                    DataType::LargeUtf8 => {
                        for value in array
                            .as_any()
                            .downcast_ref::<LargeStringArray>()
                            .unwrap()
                            .iter()
                            .flatten()
                        {
                            append(value.as_bytes());
                        }
                    }
                    DataType::Utf8View => {
                        for value in array
                            .as_any()
                            .downcast_ref::<StringViewArray>()
                            .unwrap()
                            .iter()
                            .flatten()
                        {
                            append(value.as_bytes());
                        }
                    }
                    DataType::Binary => {
                        for value in array
                            .as_any()
                            .downcast_ref::<BinaryArray>()
                            .unwrap()
                            .iter()
                            .flatten()
                        {
                            append(value);
                        }
                    }
                    DataType::LargeBinary => {
                        for value in array
                            .as_any()
                            .downcast_ref::<LargeBinaryArray>()
                            .unwrap()
                            .iter()
                            .flatten()
                        {
                            append(value);
                        }
                    }
                    DataType::BinaryView => {
                        for value in array
                            .as_any()
                            .downcast_ref::<BinaryViewArray>()
                            .unwrap()
                            .iter()
                            .flatten()
                        {
                            append(value);
                        }
                    }
                    DataType::FixedSizeBinary(_) => {
                        for value in array
                            .as_any()
                            .downcast_ref::<FixedSizeBinaryArray>()
                            .unwrap()
                            .iter()
                            .flatten()
                        {
                            append(value);
                        }
                    }
                    other => {
                        anyhow::bail!("identify expects Arrow strings or binary, got {other:?}")
                    }
                }
                let meta = serde_json::to_vec(&serde_json::json!({"n":array.len()}))?;
                let response = self.call(1, &meta, &keys)?;
                anyhow::ensure!(
                    response.len() == array.len() * 16,
                    "invalid canonical ID reply"
                );
                Ok((response, keys))
            })
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok((PyBytes::new(py, &response), PyBytes::new(py, &keys)))
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| {
            if let Some(stream) = self.stream.lock().unwrap().take() {
                let _ = stream.shutdown(Shutdown::Both);
            }
        });
    }
}

#[pyfunction]
fn run(py: Python<'_>, path: std::path::PathBuf, memory: u64, capacity: u64) -> PyResult<()> {
    py.detach(|| super::owner::run(&path, memory, capacity))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Client>()?;
    module.add("PROTOCOL", super::PROTOCOL)?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}

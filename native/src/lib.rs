//! Native kernels and shared local runtimes for relflow.
pub mod cache;

#[cfg(feature = "python")]
#[pyo3::pymodule]
fn native(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    let cache = pyo3::types::PyModule::new(module.py(), "cache")?;
    cache::python::register(&cache)?;
    module.add_submodule(&cache)
}

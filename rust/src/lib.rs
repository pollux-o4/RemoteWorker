//! RemoteWorker modular monolith; Python objects stay at the adapter boundary.
mod artifacts;
mod boundary;
mod cache;
mod encoder_policy;
mod encoder_runtime;
mod execution;
mod graph;
mod preparation;
mod profiles;
mod protocol;
mod request;
mod workflows;

pub use protocol::RequestId;
use pyo3::prelude::*;
use serde_json::Value;

fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let text: String = value
        .py()
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&text).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}
fn to_python(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("json")?
        .call_method1("loads", (value.to_string(),))?
        .unbind())
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    protocol::register(m)?;
    artifacts::register(m)?;
    cache::register(m)?;
    preparation::register(m)?;
    graph::register(m)?;
    execution::register(m)?;
    profiles::register(m)?;
    workflows::register(m)?;
    boundary::register(m)?;
    request::register(m)?;
    encoder_runtime::register(m)?;
    encoder_policy::register(m)?;
    m.add("ARCHITECTURE", "modular-monolith")?;
    Ok(())
}

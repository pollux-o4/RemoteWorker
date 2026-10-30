//! VAE domain: typed media policy, durable request storage and lossless envelopes.
mod codec;
mod execution;
mod mux;
mod policy;
mod store;
use pyo3::prelude::*;
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    codec::register(m)?;
    store::register(m)?;
    policy::register(m)?;
    mux::register(m)?;
    execution::register(m)?;
    Ok(())
}

//! Preparation admission and execution ownership; host calls stay at the FFI boundary.
mod actions;
#[path = "execution/gate.rs"]
mod admission;
mod cache;
mod coordinator;
mod handoff;
mod host;
mod integration;
mod preparation;
mod previews;
mod resolver;
mod slots;

use admission::{GateLease, gate};
use handoff::{handoff_signature, handoff_unpack};
use pyo3::{prelude::*, types::PyModule};
use resolver::Resolver;
use slots::memory_pressure;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    coordinator::register(m)?;
    actions::register(m)?;
    integration::register(m)?;
    previews::register(m)?;
    host::register(m)?;
    admission::register(m)?;
    slots::register(m)?;
    handoff::register(m)?;
    preparation::register(m)?;
    resolver::register(m)?;
    cache::register(m)?;
    Ok(())
}

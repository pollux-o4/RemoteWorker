use pyo3::{exceptions::*, prelude::*};
use std::{
    collections::HashMap,
    sync::{Arc, Condvar, Mutex, OnceLock},
    time::Duration,
};
#[derive(Default)]
struct GateStatus {
    busy: bool,
    foreground: usize,
}
#[derive(Default)]
struct GateInner {
    status: Mutex<GateStatus>,
    changed: Condvar,
}
#[pyclass]
pub(super) struct PeerGate {
    inner: Arc<GateInner>,
}
#[pymethods]
impl PeerGate {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(GateInner::default()),
        }
    }
    #[getter]
    fn busy(&self) -> bool {
        self.inner.status.lock().unwrap().busy
    }
    #[getter]
    fn foreground(&self) -> usize {
        self.inner.status.lock().unwrap().foreground
    }
    #[pyo3(signature=(speculative=false,check_wait=None))]
    pub(super) fn enter(&self, speculative: bool, check_wait: Option<Py<PyAny>>) -> GateLease {
        GateLease {
            inner: self.inner.clone(),
            speculative,
            check_wait,
            held: false,
        }
    }
}
#[pyclass]
pub(super) struct GateLease {
    inner: Arc<GateInner>,
    speculative: bool,
    check_wait: Option<Py<PyAny>>,
    held: bool,
}
#[pymethods]
impl GateLease {
    pub(super) fn __enter__(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.held {
            return Err(PyValueError::new_err("Peer gate lease already entered"));
        }
        if !self.speculative {
            self.inner.status.lock().unwrap().foreground += 1;
        }
        let result = self.acquire(py);
        if !self.speculative {
            self.inner.status.lock().unwrap().foreground -= 1;
        }
        if result.is_err() {
            self.inner.changed.notify_all();
        }
        result
    }
    fn __exit__(
        &mut self,
        _kind: &Bound<'_, PyAny>,
        _error: &Bound<'_, PyAny>,
        _trace: &Bound<'_, PyAny>,
    ) -> bool {
        self.release();
        false
    }
}
impl GateLease {
    fn acquire(&mut self, py: Python<'_>) -> PyResult<()> {
        loop {
            if let Some(check) = &self.check_wait {
                check.call0(py)?;
            }
            let mut state = self.inner.status.lock().unwrap();
            if !state.busy && (!self.speculative || state.foreground == 0) {
                state.busy = true;
                self.held = true;
                return Ok(());
            }
            drop(state);
            let inner = self.inner.clone();
            let speculative = self.speculative;
            py.detach(move || {
                let state = inner.status.lock().unwrap();
                if state.busy || (speculative && state.foreground > 0) {
                    drop(
                        inner
                            .changed
                            .wait_timeout(state, Duration::from_millis(250))
                            .unwrap(),
                    );
                }
            });
        }
    }
    pub(super) fn release(&mut self) {
        if self.held {
            self.inner.status.lock().unwrap().busy = false;
            self.held = false;
            self.inner.changed.notify_all();
        }
    }
}
impl Drop for GateLease {
    fn drop(&mut self) {
        self.release();
    }
}
static GATES: OnceLock<Mutex<HashMap<String, Arc<GateInner>>>> = OnceLock::new();
#[pyfunction]
pub(super) fn gate(url: &str) -> PeerGate {
    let mut gates = GATES.get_or_init(Default::default).lock().unwrap();
    PeerGate {
        inner: gates
            .entry(url.trim_end_matches('/').into())
            .or_default()
            .clone(),
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PeerGate>()?;
    m.add_function(wrap_pyfunction!(gate, m)?)?;
    Ok(())
}

use pyo3::{exceptions::PyValueError, prelude::*, types::*};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Local,
    Remote,
    Ready,
    Restoring,
    Failed,
}
impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
            Self::Ready => "ready",
            Self::Restoring => "restoring",
            Self::Failed => "failed",
        }
    }
}
#[pyclass]
struct Job {
    #[pyo3(get)]
    prompt_id: String,
    #[pyo3(get)]
    executor: Py<PyAny>,
    #[pyo3(get)]
    cancel: Py<PyAny>,
    phase: Phase,
    #[pyo3(get, set)]
    extra: Py<PyAny>,
    #[pyo3(get, set)]
    outputs: Py<PyAny>,
    #[pyo3(get, set)]
    previews: Py<PyAny>,
    #[pyo3(get, set)]
    preview_nodes: Py<PyAny>,
}
#[pymethods]
impl Job {
    #[new]
    fn new(py: Python<'_>, prompt_id: String, executor: Py<PyAny>) -> PyResult<Self> {
        Ok(Self {
            prompt_id,
            executor,
            cancel: py.import("threading")?.getattr("Event")?.call0()?.unbind(),
            phase: Phase::Local,
            extra: PyDict::new(py).into_any().unbind(),
            outputs: PyList::empty(py).into_any().unbind(),
            previews: PyDict::new(py).into_any().unbind(),
            preview_nodes: PyList::empty(py).into_any().unbind(),
        })
    }
    #[getter]
    fn phase(&self) -> &str {
        self.phase.label()
    }
}
#[pyclass]
struct SlotState {
    #[pyo3(get)]
    jobs: Py<PyDict>,
    #[pyo3(get)]
    active: Option<String>,
    #[pyo3(get)]
    ready: usize,
}
#[pymethods]
impl SlotState {
    #[new]
    fn new(py: Python<'_>) -> Self {
        Self {
            jobs: PyDict::new(py).unbind(),
            active: None,
            ready: 0,
        }
    }
    fn start(&mut self, py: Python<'_>, job: Py<Job>) -> PyResult<()> {
        if self.active.is_some() {
            return Err(PyValueError::new_err("Pipeline local slot already owned"));
        }
        if self.jobs.bind(py).len() >= 2 {
            return Err(PyValueError::new_err("Pipeline capacity exceeded"));
        }
        let id = job.borrow(py).prompt_id.clone();
        if self.jobs.bind(py).contains(&id)? {
            return Err(PyValueError::new_err("Duplicate pipeline prompt"));
        }
        self.jobs.bind(py).set_item(&id, job)?;
        self.active = Some(id);
        Ok(())
    }
    fn finish(&mut self, py: Python<'_>, job: Py<Job>) -> PyResult<()> {
        let info = job.borrow(py);
        let id = info.prompt_id.clone();
        if info.phase == Phase::Ready {
            self.ready -= 1;
        }
        drop(info);
        self.jobs.bind(py).del_item(&id)?;
        if self.active.as_ref() == Some(&id) {
            self.active = None;
        }
        Ok(())
    }
    fn suspend(&mut self, py: Python<'_>, job: Py<Job>) -> PyResult<()> {
        let mut job = job.borrow_mut(py);
        if job.phase != Phase::Local {
            return Err(PyValueError::new_err(
                "Pipeline job must own the local slot before remote IO",
            ));
        }
        job.phase = Phase::Remote;
        if self.active.as_ref() == Some(&job.prompt_id) {
            self.active = None;
        }
        Ok(())
    }
    fn completed(&mut self, py: Python<'_>, job: Py<Job>) -> PyResult<()> {
        let mut job = job.borrow_mut(py);
        if job.phase != Phase::Remote {
            return Err(PyValueError::new_err(
                "Pipeline completion requires a remote job",
            ));
        }
        job.phase = Phase::Ready;
        self.ready += 1;
        Ok(())
    }
    fn reenter(&mut self, py: Python<'_>, job: Py<Job>) -> PyResult<()> {
        if self.active.is_some() {
            return Err(PyValueError::new_err("Pipeline local slot already owned"));
        }
        let mut job = job.borrow_mut(py);
        if job.phase != Phase::Ready {
            return Err(PyValueError::new_err(
                "Pipeline reentry requires a ready job",
            ));
        }
        job.phase = Phase::Restoring;
        self.ready -= 1;
        Ok(())
    }
    fn restore_failed(&mut self, py: Python<'_>, job: Py<Job>) -> PyResult<()> {
        let mut job = job.borrow_mut(py);
        if job.phase != Phase::Restoring {
            return Err(PyValueError::new_err(
                "Pipeline restore failure requires reentry",
            ));
        }
        job.phase = Phase::Failed;
        Ok(())
    }
    fn resume(&mut self, py: Python<'_>, job: Py<Job>) -> PyResult<()> {
        if self.active.is_some() {
            return Err(PyValueError::new_err("Pipeline local slot already owned"));
        }
        let mut job = job.borrow_mut(py);
        match job.phase {
            Phase::Ready => self.ready -= 1,
            Phase::Restoring => (),
            _ => {
                return Err(PyValueError::new_err(
                    "Pipeline resume requires a ready job",
                ));
            }
        }
        job.phase = Phase::Local;
        self.active = Some(job.prompt_id.clone());
        Ok(())
    }
    fn status(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        result.set_item("enabled", true)?;
        result.set_item("capacity", 2)?;
        result.set_item("local_prompt_id", &self.active)?;
        let jobs = PyList::empty(py);
        for (_, value) in self.jobs.bind(py).iter() {
            let job = value.extract::<PyRef<'_, Job>>()?;
            let row = PyDict::new(py);
            row.set_item("prompt_id", &job.prompt_id)?;
            row.set_item("phase", job.phase.label())?;
            row.set_item(
                "preview_nodes",
                job.preview_nodes.bind(py).call_method0("copy")?,
            )?;
            jobs.append(row)?;
        }
        result.set_item("jobs", jobs)?;
        Ok(result.unbind())
    }
}
#[pyfunction]
pub(super) fn validate_preparation(role: &str, kind: Option<&str>) -> PyResult<()> {
    match role {
        "encoder" => Ok(()),
        "decoder" if matches!(kind, Some("image" | "audio")) => Ok(()),
        "decoder" => Err(PyValueError::new_err(
            "Decoder preparation requires a media kind",
        )),
        _ => Err(PyValueError::new_err("Unknown preparation role")),
    }
}
#[pyfunction]
pub(super) fn memory_pressure(available: Option<u64>, now: f64, last: f64) -> bool {
    available.is_some_and(|n| n < 2 * 1024 * 1024 * 1024) && now - last > 30.0
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Job>()?;
    m.add_class::<SlotState>()?;
    m.add_function(wrap_pyfunction!(validate_preparation, m)?)?;
    m.add_function(wrap_pyfunction!(memory_pressure, m)?)?;
    Ok(())
}

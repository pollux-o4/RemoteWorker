//! Preparation admission and execution ownership; host calls stay at the FFI boundary.
mod actions;
mod coordinator;
mod integration;
mod previews;

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use std::{
    collections::{HashMap, HashSet},
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
struct PeerGate {
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
    fn enter(&self, speculative: bool, check_wait: Option<Py<PyAny>>) -> GateLease {
        GateLease {
            inner: self.inner.clone(),
            speculative,
            check_wait,
            held: false,
        }
    }
}
#[pyclass]
struct GateLease {
    inner: Arc<GateInner>,
    speculative: bool,
    check_wait: Option<Py<PyAny>>,
    held: bool,
}
#[pymethods]
impl GateLease {
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<()> {
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
    fn release(&mut self) {
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
fn gate(url: &str) -> PeerGate {
    let mut gates = GATES.get_or_init(Default::default).lock().unwrap();
    PeerGate {
        inner: gates
            .entry(url.trim_end_matches('/').into())
            .or_default()
            .clone(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Local,
    Remote,
    Ready,
}
impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Remote => "remote",
            Self::Ready => "ready",
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
    fn resume(&mut self, py: Python<'_>, job: Py<Job>) -> PyResult<()> {
        if self.active.is_some() {
            return Err(PyValueError::new_err("Pipeline local slot already owned"));
        }
        let mut job = job.borrow_mut(py);
        if job.phase != Phase::Ready {
            return Err(PyValueError::new_err(
                "Pipeline resume requires a ready job",
            ));
        }
        job.phase = Phase::Local;
        self.ready -= 1;
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
fn validate_preparation(role: &str, kind: Option<&str>) -> PyResult<()> {
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
fn memory_pressure(available: Option<u64>, now: f64, last: f64) -> bool {
    available.is_some_and(|n| n < 2 * 1024 * 1024 * 1024) && now - last > 30.0
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    coordinator::register(m)?;
    actions::register(m)?;
    integration::register(m)?;
    previews::register(m)?;
    m.add_class::<PeerGate>()?;
    m.add_function(wrap_pyfunction!(gate, m)?)?;
    m.add_class::<Job>()?;
    m.add_class::<SlotState>()?;
    m.add_function(wrap_pyfunction!(validate_preparation, m)?)?;
    m.add_function(wrap_pyfunction!(memory_pressure, m)?)?;
    m.add_function(wrap_pyfunction!(handoff_signature, m)?)?;
    m.add_function(wrap_pyfunction!(handoff_pack, m)?)?;
    m.add_function(wrap_pyfunction!(handoff_unpack, m)?)?;
    m.add_function(wrap_pyfunction!(memory_snapshot, m)?)?;
    m.add_class::<Preparation>()?;
    m.add_function(wrap_pyfunction!(discover_preparation, m)?)?;
    m.add_function(wrap_pyfunction!(preparation_encoders, m)?)?;
    m.add_function(wrap_pyfunction!(preparation_decoders, m)?)?;
    m.add_function(wrap_pyfunction!(sampling_in, m)?)?;
    m.add_function(wrap_pyfunction!(prep_observe_event, m)?)?;
    m.add_class::<Resolver>()?;
    m.add("UnsupportedInput", m.py().get_type::<UnsupportedInput>())?;
    m.add_class::<WeakModelCache>()?;
    m.add_function(wrap_pyfunction!(model_cache_eligible, m)?)?;
    Ok(())
}

use pyo3::types::{PyFloat, PyInt, PyString};
use std::collections::VecDeque;
enum WeakValue {
    Primitive(Py<PyAny>),
    Reference(Py<PyAny>),
    List(Vec<WeakValue>),
    Tuple(Vec<WeakValue>),
}
impl WeakValue {
    fn pack(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            let mut parts = Vec::new();
            for part in value.try_iter()? {
                parts.push(Self::pack(&part?)?);
            }
            return Ok(if value.is_instance_of::<PyList>() {
                Self::List(parts)
            } else {
                Self::Tuple(parts)
            });
        }
        if value.is_none()
            || value.is_instance_of::<PyString>()
            || value.is_instance_of::<PyInt>()
            || value.is_instance_of::<PyFloat>()
        {
            return Ok(Self::Primitive(value.clone().unbind()));
        }
        Ok(Self::Reference(
            value
                .py()
                .import("weakref")?
                .getattr("ref")?
                .call1((value,))?
                .unbind(),
        ))
    }
    fn unpack(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match self {
            Self::Primitive(value) => Ok(Some(value.clone_ref(py))),
            Self::Reference(reference) => {
                let value = reference.call0(py)?;
                Ok(if value.is_none(py) { None } else { Some(value) })
            }
            Self::List(parts) | Self::Tuple(parts) => {
                let mut result = Vec::new();
                for part in parts {
                    let Some(value) = part.unpack(py)? else {
                        return Ok(None);
                    };
                    result.push(value);
                }
                Ok(Some(match self {
                    Self::List(_) => PyList::new(py, result)?.into_any().unbind(),
                    _ => PyTuple::new(py, result)?.into_any().unbind(),
                }))
            }
        }
    }
}
#[pyclass]
struct WeakModelCache {
    entries: VecDeque<(Py<PyAny>, WeakValue)>,
    #[pyo3(get, set)]
    hits: u64,
}
#[pymethods]
impl WeakModelCache {
    #[new]
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            hits: 0,
        }
    }
    fn clear(&mut self) {
        self.entries.clear();
    }
    fn remember(&mut self, key: &Bound<'_, PyAny>, outputs: &Bound<'_, PyAny>) -> PyResult<()> {
        let packed = match WeakValue::pack(outputs) {
            Ok(value) => value,
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyTypeError>(key.py()) => {
                return Ok(());
            }
            Err(err) => return Err(err),
        };
        if let Some(index) = self.index(key)? {
            self.entries.remove(index);
        }
        self.entries.push_back((key.clone().unbind(), packed));
        while self.entries.len() > 16 {
            self.entries.pop_front();
        }
        Ok(())
    }
    fn recall(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let Some(index) = self.index(key)? else {
            return Ok(None);
        };
        let result = self.entries[index].1.unpack(key.py())?;
        if result.is_none() {
            self.entries.remove(index);
        }
        Ok(result)
    }
}
impl WeakModelCache {
    fn index(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        key.hash()?;
        for (index, (stored, _)) in self.entries.iter().enumerate() {
            if stored.bind(key.py()).eq(key)? {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }
}
#[pyfunction]
#[pyo3(signature=(cls,value=None))]
fn model_cache_eligible(
    cls: &Bound<'_, PyAny>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    if cls.is_none() {
        return Ok(false);
    }
    let returns = cls
        .getattr("RETURN_TYPES")
        .unwrap_or_else(|_| PyTuple::empty(cls.py()).into_any());
    if !returns.contains("MODEL")? {
        return Ok(false);
    }
    Ok(!value.is_some_and(|v| {
        v.is_truthy().unwrap_or(false)
            && v.getattr("ui").and_then(|v| v.is_truthy()).unwrap_or(false)
    }))
}

pyo3::create_exception!(_native, UnsupportedInput, PyValueError);
const SAFE_TYPES: &[&str] = &[
    "LoadImage",
    "ResizeImageMaskNode",
    "ResolutionSelector",
    "GetImageSize",
    "ImageScaleToTotalPixels",
    "ImageScale",
    "ImageScaleBy",
    "ImageFromBatch",
    "ImageBatch",
    "PrimitiveFloat",
    "PrimitiveInt",
    "PrimitiveBoolean",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "RequestConditionInput",
    "GenRelInput",
    "ComfyMathExpression",
    "RemoteVAESelector",
];
#[pyclass]
struct Resolver {
    graph: Py<PyDict>,
    classes: Py<PyDict>,
    outputs: Py<PyDict>,
    visiting: HashSet<String>,
    check_cancelled: Py<PyAny>,
}
#[pymethods]
impl Resolver {
    #[new]
    fn new(
        py: Python<'_>,
        graph: Py<PyDict>,
        classes: Py<PyDict>,
        check_cancelled: Py<PyAny>,
    ) -> Self {
        Self {
            graph,
            classes,
            outputs: PyDict::new(py).unbind(),
            visiting: HashSet::new(),
            check_cancelled,
        }
    }
    #[pyo3(signature=(node_id,exclude=None))]
    fn inputs(
        &mut self,
        py: Python<'_>,
        node_id: &str,
        exclude: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        self.resolve_inputs(py, node_id, exclude)
    }
    fn value(&mut self, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.resolve_value(value)
    }
}
fn required<'py>(values: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    values
        .get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))
}
impl Resolver {
    fn resolve_inputs(
        &mut self,
        py: Python<'_>,
        node_id: &str,
        exclude: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        let node = self
            .graph
            .bind(py)
            .get_item(node_id)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(node_id.to_owned()))?;
        let inputs = node
            .call_method1("get", ("inputs", PyDict::new(py)))?
            .cast_into::<PyDict>()?;
        let result = PyDict::new(py);
        for (key, value) in inputs.iter() {
            if exclude.is_some_and(|exclude| exclude.contains(&key).unwrap_or(false)) {
                continue;
            }
            result.set_item(key, self.resolve_value(&value)?)?;
        }
        Ok(result.unbind())
    }
    fn resolve_value(&mut self, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = value.py();
        let Ok(link) = value.cast::<PyList>() else {
            return Ok(value.clone().unbind());
        };
        if link.len() != 2 {
            return Ok(value.clone().unbind());
        }
        let key: String = link.get_item(0)?.str()?.extract()?;
        let Some(node) = self.graph.bind(py).get_item(&key)? else {
            return Ok(value.clone().unbind());
        };
        let slot = link.get_item(1)?;
        if !self.outputs.bind(py).contains(&key)? {
            let kind: String = node.get_item("class_type")?.extract()?;
            let cls = self.classes.bind(py).get_item(&kind)?;
            if !SAFE_TYPES.contains(&kind.as_str()) || cls.is_none() {
                return Err(UnsupportedInput::new_err(format!(
                    "upstream_not_cpu_safe: {kind}"
                )));
            }
            if self.visiting.contains(&key) || self.visiting.len() > 64 {
                return Err(UnsupportedInput::new_err("upstream_cycle_or_depth"));
            }
            self.visiting.insert(key.clone());
            let result = self.execute(py, &key, &kind, cls.unwrap());
            self.visiting.remove(&key);
            result?;
        }
        Ok(self
            .outputs
            .bind(py)
            .get_item(&key)?
            .unwrap()
            .get_item(slot)?
            .unbind())
    }
    fn execute(
        &mut self,
        py: Python<'_>,
        key: &str,
        kind: &str,
        cls: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.check_cancelled.call0(py)?;
        let mut values = self.resolve_inputs(py, key, None)?;
        if kind == "ComfyMathExpression" {
            let mapped = PyDict::new(py);
            mapped.set_item("expression", required(values.bind(py), "expression")?)?;
            let nested = PyDict::new(py);
            for (k, v) in values.bind(py).iter() {
                let k: String = k.extract()?;
                if let Some(k) = k.strip_prefix("values.") {
                    nested.set_item(k, v)?;
                }
            }
            mapped.set_item("values", nested)?;
            values = mapped.unbind();
        }
        if kind == "ResizeImageMaskNode" {
            let mapped = PyDict::new(py);
            let nested = PyDict::new(py);
            for field in ["input", "scale_method"] {
                mapped.set_item(field, required(values.bind(py), field)?)?;
            }
            nested.set_item("resize_type", required(values.bind(py), "resize_type")?)?;
            for (k, v) in values.bind(py).iter() {
                let k: String = k.extract()?;
                if let Some(k) = k.strip_prefix("resize_type.") {
                    nested.set_item(k, v)?;
                }
            }
            mapped.set_item("resize_type", nested)?;
            values = mapped.unbind();
        }
        let mut result = if kind == "GetImageSize" {
            let shape = required(values.bind(py), "image")?.getattr("shape")?;
            PyTuple::new(
                py,
                [shape.get_item(2)?, shape.get_item(1)?, shape.get_item(0)?],
            )?
            .into_any()
        } else {
            let function = cls
                .getattr("FUNCTION")
                .map(|s| s.extract::<String>())
                .unwrap_or_else(|_| Ok("execute".into()))?;
            cls.call0()?
                .getattr(function.as_str())?
                .call((), Some(values.bind(py)))?
        };
        if result.hasattr("result")? {
            result = result.getattr("result")?;
        }
        if let Ok(mapping) = result.cast::<PyDict>() {
            if mapping
                .get_item("expand")?
                .is_some_and(|v| v.is_truthy().unwrap_or(false))
            {
                return Err(UnsupportedInput::new_err("upstream_expansion"));
            }
            result = mapping
                .get_item("result")?
                .unwrap_or_else(|| py.None().into_bound(py));
        }
        if !result.is_instance_of::<PyTuple>() && !result.is_instance_of::<PyList>() {
            return Err(UnsupportedInput::new_err(format!(
                "upstream_result: {kind}"
            )));
        }
        self.outputs.bind(py).set_item(key, result)?;
        self.check_cancelled.call0(py)?;
        Ok(())
    }
}

fn digest(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    py.import("hashlib")?
        .call_method1("sha256", (value,))?
        .call_method0("hexdigest")?
        .extract()
}
fn signature_value(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    if value.hasattr("detach")? {
        let tensor = value
            .call_method0("detach")?
            .call_method0("cpu")?
            .call_method0("contiguous")?;
        let bytes = tensor
            .call_method1("view", (py.import("torch")?.getattr("uint8")?,))?
            .call_method0("numpy")?
            .call_method0("tobytes")?;
        let result = PyDict::new(py);
        result.set_item(
            "shape",
            py.import("builtins")?
                .getattr("list")?
                .call1((tensor.getattr("shape")?,))?,
        )?;
        result.set_item("dtype", tensor.getattr("dtype")?.str()?)?;
        result.set_item("sha256", digest(py, &bytes)?)?;
        return Ok(result.into_any().unbind());
    }
    if let Ok(values) = value.cast::<PyDict>() {
        let result = PyDict::new(py);
        for (k, v) in values.iter() {
            result.set_item(k, signature_value(&v)?)?;
        }
        return Ok(result.into_any().unbind());
    }
    if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
        let result = PyList::empty(py);
        for v in value.try_iter()? {
            result.append(signature_value(&v?)?)?;
        }
        return Ok(result.into_any().unbind());
    }
    Ok(value.clone().unbind())
}
#[pyfunction]
fn handoff_signature(values: &Bound<'_, PyDict>) -> PyResult<String> {
    let py = values.py();
    let filtered = PyDict::new(py);
    for (k, v) in values.iter() {
        let name: String = k.extract()?;
        if !["self", "unique_id", "extra_pnginfo", "timeout_seconds"].contains(&name.as_str()) {
            filtered.set_item(k, v)?;
        }
    }
    let normalized = signature_value(filtered.as_any())?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("sort_keys", true)?;
    let encoded = py
        .import("json")?
        .call_method("dumps", (normalized,), Some(&kwargs))?
        .call_method0("encode")?;
    digest(py, &encoded)
}
fn tensor_bytes(value: &Bound<'_, PyAny>) -> PyResult<u128> {
    if value.hasattr("numel")? {
        if value
            .getattr("is_nested")
            .and_then(|v| v.is_truthy())
            .unwrap_or(false)
        {
            let mut bytes = 0;
            for v in value.call_method0("unbind")?.try_iter()? {
                bytes += tensor_bytes(&v?)?;
            }
            return Ok(bytes);
        }
        let elements: u128 = value.call_method0("numel")?.extract()?;
        let size: u128 = value.call_method0("element_size")?.extract()?;
        return Ok(elements * size);
    }
    let mut size = 0;
    if let Ok(values) = value.cast::<PyDict>() {
        for (_, v) in values.iter() {
            size += tensor_bytes(&v)?;
        }
    } else if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
        for v in value.try_iter()? {
            size += tensor_bytes(&v?)?;
        }
    }
    Ok(size)
}
#[pyfunction]
#[pyo3(signature=(result,max_bytes=None))]
fn handoff_pack(result: &Bound<'_, PyAny>, max_bytes: Option<u128>) -> PyResult<Option<Py<PyAny>>> {
    let py = result.py();
    if let Some(max) = max_bytes {
        if tensor_bytes(result)? > max {
            return Ok(None);
        }
    }
    let buffer = py.import("io")?.getattr("BytesIO")?.call0()?;
    py.import("torch")?
        .call_method1("save", (result, &buffer))?;
    let size: u128 = buffer.call_method0("tell")?.extract()?;
    if max_bytes.is_some_and(|max| size > max) {
        return Ok(None);
    }
    Ok(Some(buffer.call_method0("getvalue")?.unbind()))
}
#[pyfunction]
fn handoff_unpack(blob: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = blob.py();
    let buffer = py.import("io")?.getattr("BytesIO")?.call1((blob,))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("map_location", "cpu")?;
    kwargs.set_item("weights_only", false)?;
    Ok(py
        .import("torch")?
        .call_method("load", (buffer,), Some(&kwargs))?
        .unbind())
}
#[pyfunction]
fn memory_snapshot(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return Ok(result.unbind());
    };
    let values: HashMap<_, _> = text
        .lines()
        .filter_map(|line| {
            let (name, raw) = line.split_once(':')?;
            Some((
                name,
                raw.split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)?,
            ))
        })
        .collect();
    if let (Some(available), Some(swap), Some(free)) = (
        values.get("MemAvailable"),
        values.get("SwapTotal"),
        values.get("SwapFree"),
    ) {
        result.set_item("available", available)?;
        result.set_item("swap_used", swap.saturating_sub(*free))?;
    }
    Ok(result.unbind())
}

#[derive(Clone)]
enum PreparationRole {
    Encoder(Option<String>),
    ImageDecoder,
    AudioDecoder,
}
#[pyclass(frozen)]
struct Preparation {
    role: PreparationRole,
    #[pyo3(get)]
    executor: Option<Py<PyAny>>,
    #[pyo3(get)]
    inputs: Py<PyAny>,
}
#[pymethods]
impl Preparation {
    #[new]
    #[pyo3(signature=(role,kind=None,executor=None,inputs=None))]
    fn new(
        py: Python<'_>,
        role: &str,
        kind: Option<String>,
        executor: Option<Py<PyAny>>,
        inputs: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        validate_preparation(role, kind.as_deref())?;
        let role = match role {
            "encoder" => PreparationRole::Encoder(kind),
            "decoder" if kind.as_deref() == Some("image") => PreparationRole::ImageDecoder,
            _ => PreparationRole::AudioDecoder,
        };
        Ok(Self {
            role,
            executor,
            inputs: match inputs {
                Some(v) => v,
                None => py.import("builtins")?.getattr("dict")?.unbind(),
            },
        })
    }
    #[getter]
    fn role(&self) -> &str {
        match self.role {
            PreparationRole::Encoder(_) => "encoder",
            _ => "decoder",
        }
    }
    #[getter]
    fn kind(&self) -> Option<String> {
        match &self.role {
            PreparationRole::Encoder(kind) => kind.clone(),
            PreparationRole::ImageDecoder => Some("image".into()),
            PreparationRole::AudioDecoder => Some("audio".into()),
        }
    }
}
#[pyfunction]
fn discover_preparation(classes: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(classes.py());
    for (name, cls) in classes.iter() {
        if let Ok(contract) = cls.getattr("REMOTE_PREPARATION") {
            if contract.is_instance_of::<Preparation>() {
                result.set_item(name, (cls, contract))?;
            }
        }
    }
    Ok(result.unbind())
}
#[pyfunction]
fn preparation_encoders(registered: &Bound<'_, PyDict>) -> PyResult<HashSet<String>> {
    let mut result = HashSet::new();
    for (name, value) in registered.iter() {
        let spec = value.get_item(1)?.extract::<PyRef<'_, Preparation>>()?;
        if matches!(spec.role, PreparationRole::Encoder(_)) {
            result.insert(name.extract()?);
        }
    }
    Ok(result)
}
#[pyfunction]
fn preparation_decoders(registered: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut result = HashMap::new();
    for (name, value) in registered.iter() {
        let spec = value.get_item(1)?.extract::<PyRef<'_, Preparation>>()?;
        if !matches!(spec.role, PreparationRole::Encoder(_)) {
            result.insert(name.extract()?, spec.kind().unwrap());
        }
    }
    Ok(result)
}
#[pyfunction]
fn sampling_in(registry: &Bound<'_, PyAny>, prompt_id: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !registry.getattr("prompt_id")?.eq(prompt_id)? {
        return Ok(false);
    }
    for (node_id, entry) in registry
        .getattr("nodes")?
        .call_method0("copy")?
        .cast::<PyDict>()?
        .iter()
    {
        let state = entry.call_method1("get", ("state",))?;
        let state = state.getattr("value").unwrap_or(state);
        if state.extract::<String>().is_ok_and(|s| s == "running") {
            let node = registry
                .getattr("dynprompt")?
                .call_method1("get_node", (node_id,))?;
            if node
                .call_method1("get", ("class_type", ""))?
                .extract::<String>()?
                .contains("Sampler")
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
#[pyfunction]
fn prep_observe_event(
    coordinator: &Bound<'_, PyAny>,
    event: &str,
    data: &Bound<'_, PyAny>,
    read_registry: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if event == "progress_state" && data.is_instance_of::<PyDict>() {
        let registry = read_registry.call0()?;
        let job = data.call_method1("get", ("prompt_id",))?;
        if registry.getattr("prompt_id")?.eq(&job)? {
            for (key, value) in data
                .call_method1("get", ("nodes", PyDict::new(data.py())))?
                .cast::<PyDict>()?
                .iter()
            {
                if value
                    .call_method1("get", ("state",))?
                    .extract::<String>()
                    .unwrap_or_default()
                    != "running"
                {
                    continue;
                }
                let node = registry
                    .getattr("dynprompt")?
                    .call_method1("get_node", (&key,))?;
                if node
                    .call_method1("get", ("class_type", ""))?
                    .extract::<String>()?
                    .contains("Sampler")
                {
                    coordinator
                        .getattr("scheduler")?
                        .getattr("sampling")?
                        .call_method1(
                            "update",
                            (
                                &job,
                                key,
                                value.call_method1("get", ("value", 0))?,
                                value.call_method1("get", ("max", 1))?,
                            ),
                        )?;
                }
            }
        }
        coordinator.getattr("wake")?.call_method0("set")?;
    } else if [
        "status",
        "executing",
        "execution_start",
        "execution_success",
        "execution_error",
        "execution_interrupted",
    ]
    .contains(&event)
    {
        coordinator.getattr("wake")?.call_method0("set")?;
    }
    Ok(())
}

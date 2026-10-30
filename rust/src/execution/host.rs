//! Comfy host scheduling; Python supplies only framework constructors and I/O callbacks.
use super::actions::with_context;
use pyo3::{
    exceptions::{PyException, PyValueError},
    prelude::*,
    types::{PyCFunction, PyDict, PyList, PySlice, PyTuple},
};
use std::{collections::HashSet, time::Duration};

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Admission {
    #[default]
    Open,
    RamPressure,
}
#[pyclass]
struct HostPolicy {
    admission: Admission,
    last_gc: f64,
    pending_cancel: HashSet<String>,
    #[pyo3(get)]
    stopping: bool,
}
#[pymethods]
impl HostPolicy {
    #[new]
    fn new() -> Self {
        Self {
            admission: Admission::Open,
            last_gc: 0.0,
            pending_cancel: HashSet::new(),
            stopping: false,
        }
    }
    #[getter]
    fn admission(&self) -> Option<&str> {
        match self.admission {
            Admission::Open => None,
            Admission::RamPressure => Some("ram_pressure"),
        }
    }
    fn pending(&self, prompt: &str) -> bool {
        self.pending_cancel.contains(prompt)
    }
}
fn policy<'py>(host: &Bound<'py, PyAny>) -> PyResult<Py<HostPolicy>> {
    Ok(host.getattr("policy")?.extract()?)
}
fn now(source: &Bound<'_, PyModule>, clock: &str) -> PyResult<f64> {
    source.getattr("time")?.call_method0(clock)?.extract()
}
fn log(source: &Bound<'_, PyModule>, message: &str, error: &PyErr) -> PyResult<()> {
    let py = source.py();
    let kw = PyDict::new(py);
    kw.set_item(
        "exc_info",
        (error.get_type(py), error.value(py), error.traceback(py)),
    )?;
    source
        .getattr("logging")?
        .call_method1("getLogger", (source.name()?,))?
        .call_method("error", (message,), Some(&kw))?;
    Ok(())
}

#[pyfunction]
fn host_status(host: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let result = host
        .getattr("slots")?
        .call_method0("status")?
        .cast_into::<PyDict>()?;
    result.set_item("admission", policy(host)?.borrow(host.py()).admission())?;
    result.set_item("model_reuse_hits", host.getattr("reuse")?.getattr("hits")?)?;
    Ok(result.unbind())
}
#[pyfunction]
fn host_share_models(
    host: &Bound<'_, PyAny>,
    executor: &Bound<'_, PyAny>,
    contains_unequal: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let cache = executor.getattr("caches")?.getattr("outputs")?;
    if !cache
        .getattr("initialized")
        .and_then(|v| v.is_truthy())
        .unwrap_or(false)
    {
        return Ok(());
    }
    let reuse = host.getattr("reuse")?;
    for node_id in cache
        .getattr("cache_key_set")?
        .call_method0("all_node_ids")?
        .try_iter()?
    {
        let node_id = node_id?;
        let key = cache
            .getattr("cache_key_set")?
            .call_method1("get_data_key", (&node_id,))?;
        if key.is_none() || contains_unequal.call1((&key,))?.is_truthy()? {
            continue;
        }
        let context = cache.call_method1("_build_context", (&node_id, key))?;
        if context.is_none()
            || !reuse
                .call_method1("should_cache", (&context,))?
                .is_truthy()?
        {
            continue;
        }
        let value = cache.call_method1("get_local", (node_id,))?;
        if !value.is_none()
            && reuse
                .call_method1("should_cache", (&context, &value))?
                .is_truthy()?
        {
            reuse.call_method1(
                "remember",
                (
                    (
                        context.getattr("class_type")?,
                        context.getattr("cache_key_hash")?,
                    ),
                    value.getattr("outputs")?,
                ),
            )?;
        }
    }
    Ok(())
}
#[derive(Debug, PartialEq, Eq)]
enum CacheSelection {
    Classic,
    Lru,
    None,
    RamPressure,
}
fn cache_selection(classic: bool, lru: i64, none: bool) -> CacheSelection {
    if classic {
        CacheSelection::Classic
    } else if lru > 0 {
        CacheSelection::Lru
    } else if none {
        CacheSelection::None
    } else {
        CacheSelection::RamPressure
    }
}
fn cache_budgets(total_ram: f64, override_ram: &[f64]) -> (f64, f64) {
    let ram = override_ram
        .first()
        .copied()
        .unwrap_or((total_ram * 0.10 / 1024.0).clamp(2.0, 10.0));
    let inactive = override_ram
        .get(1)
        .copied()
        .unwrap_or((total_ram / 1024.0).min(128.0));
    (ram, inactive)
}
#[pyfunction]
fn host_executor(
    host: &Bound<'_, PyAny>,
    constructor: &Bound<'_, PyAny>,
    cache_types: &Bound<'_, PyAny>,
    total_ram: f64,
) -> PyResult<Py<PyAny>> {
    if !total_ram.is_finite() || total_ram < 0.0 {
        return Err(PyValueError::new_err("Invalid Comfy total RAM"));
    }
    let py = host.py();
    let args = host.getattr("entry")?.getattr("args")?;
    let lru: i64 = args.getattr("cache_lru")?.extract()?;
    let kind = cache_selection(
        args.getattr("cache_classic")?.is_truthy()?,
        lru,
        args.getattr("cache_none")?.is_truthy()?,
    );
    let name = match kind {
        CacheSelection::Classic => "CLASSIC",
        CacheSelection::Lru => "LRU",
        CacheSelection::None => "NONE",
        CacheSelection::RamPressure => "RAM_PRESSURE",
    };
    let overrides = args.getattr("cache_ram")?;
    let overrides: Vec<f64> = if overrides.is_truthy()? {
        overrides.extract()?
    } else {
        Vec::new()
    };
    if overrides.iter().any(|n| !n.is_finite() || *n < 0.0) {
        return Err(PyValueError::new_err("Invalid Comfy RAM cache budget"));
    }
    let (ram, inactive) = cache_budgets(total_ram, &overrides);
    let kwargs = PyDict::new(py);
    let budgets = PyDict::new(py);
    budgets.set_item("lru", lru)?;
    budgets.set_item("ram", ram)?;
    budgets.set_item("ram_inactive", inactive)?;
    kwargs.set_item("cache_type", cache_types.getattr(name)?)?;
    kwargs.set_item("cache_args", budgets)?;
    kwargs.set_item("asset_manager", host.getattr("assets")?)?;
    let executor = constructor.call((host.getattr("server")?,), Some(&kwargs))?;
    host.getattr("executors")?
        .call_method1("append", (&executor,))?;
    Ok(executor.unbind())
}
#[pyfunction]
fn host_cancel(host: &Bound<'_, PyAny>, prompt_id: &str) -> PyResult<bool> {
    let py = host.py();
    let queue = host.getattr("queue")?;
    let mut running = false;
    for item in queue
        .getattr("currently_running")?
        .call_method0("values")?
        .try_iter()?
    {
        if item?.get_item(1)?.extract::<String>()? == prompt_id {
            running = true;
            break;
        }
    }
    if !running {
        return Ok(false);
    }
    let slots = host.getattr("slots")?;
    let job = slots.getattr("jobs")?.call_method1("get", (prompt_id,))?;
    if job.is_none() {
        policy(host)?
            .borrow_mut(py)
            .pending_cancel
            .insert(prompt_id.into());
        return Ok(true);
    }
    job.getattr("cancel")?.call_method0("set")?;
    if slots
        .getattr("active")?
        .extract::<Option<String>>()?
        .as_deref()
        == Some(prompt_id)
    {
        host.getattr("entry")?
            .getattr("nodes")?
            .call_method0("interrupt_processing")?;
    }
    Ok(true)
}
#[pyfunction]
fn host_interrupt(host: &Bound<'_, PyAny>, data: &Bound<'_, PyAny>) -> PyResult<()> {
    let data = data
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("Interrupt payload must be an object"))?;
    let prompt = data.get_item("prompt_id")?;
    if let Some(prompt) = prompt.as_ref().filter(|v| !v.is_none()) {
        if prompt.extract::<String>().is_err() || !prompt.is_truthy()? {
            return Err(PyValueError::new_err(
                "Interrupt prompt_id must be a non-empty string",
            ));
        }
    }
    if let Some(prompt) = prompt.filter(|v| v.is_truthy().unwrap_or(false)) {
        host.call_method1("cancel", (prompt,))?;
    } else {
        for job in host.call_method0("status")?.get_item("jobs")?.try_iter()? {
            host.call_method1("cancel", (job?.get_item("prompt_id")?,))?;
        }
    }
    Ok(())
}
#[pyfunction]
fn host_housekeeping(host: &Bound<'_, PyAny>, source: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = host.py();
    let queue = host.getattr("queue")?;
    if queue.call_method0("get_tasks_remaining")?.is_truthy()? {
        return Ok(());
    }
    let flags = queue.call_method0("get_flags")?;
    let free = flags.call_method1("get", ("free_memory", false))?;
    let models = host
        .getattr("entry")?
        .getattr("comfy")?
        .getattr("model_management")?;
    if flags
        .call_method1("get", ("unload_models", &free))?
        .is_truthy()?
    {
        models.call_method0("unload_all_models")?;
    }
    if free.is_truthy()? {
        host.getattr("reuse")?.call_method0("clear")?;
        for executor in host.getattr("executors")?.try_iter()? {
            executor?.call_method0("reset")?;
        }
    }
    if now(source, "monotonic")? - policy(host)?.borrow(py).last_gc >= 10.0 || flags.is_truthy()? {
        source.getattr("gc")?.call_method0("collect")?;
        models.call_method0("soft_empty_cache")?;
        host.getattr("entry")?
            .getattr("hook_breaker_ac10a0")?
            .call_method0("restore_functions")?;
        let assets = host.getattr("assets")?;
        assets.call_method0("queue_output_scan")?;
        assets.call_method0("resume_background_scan")?;
        policy(host)?.borrow_mut(py).last_gc = now(source, "monotonic")?;
    }
    Ok(())
}
#[pyfunction]
fn host_history_item(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    let a = value.get_item(PySlice::new(py, 0, 5, 1))?;
    let length: isize = value
        .len()?
        .try_into()
        .map_err(|_| PyValueError::new_err("History tuple too large"))?;
    let b = value.get_item(PySlice::new(py, 6, length, 1))?;
    Ok(a.call_method1("__add__", (b,))?.unbind())
}
#[pyfunction]
#[pyo3(signature=(host,executor,source,prompt_queue))]
fn host_worker_tick(
    host: &Bound<'_, PyAny>,
    executor: &Bound<'_, PyAny>,
    source: &Bound<'_, PyModule>,
    prompt_queue: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = host.py();
    if policy(host)?.borrow(py).stopping {
        return Ok(executor.clone().unbind());
    }
    if policy(host)?.borrow(py).admission == Admission::RamPressure {
        py.detach(|| std::thread::sleep(Duration::from_millis(100)));
    }
    let slots = host.getattr("slots")?;
    if slots.getattr("ready")?.is_truthy()? {
        py.detach(|| std::thread::sleep(Duration::from_millis(10)));
        return Ok(executor.clone().unbind());
    }
    with_context(&slots.getattr("lock")?, || {
        if policy(host)?.borrow(py).stopping {
            return Ok(executor.clone().unbind());
        }
        if slots.getattr("ready")?.is_truthy()? {
            return Ok(executor.clone().unbind());
        }
        let executor = if executor.is_none() {
            host.call_method0("executor")?
        } else {
            executor.clone()
        };
        if slots.getattr("jobs")?.is_truthy()? {
            let memory = source.getattr("snapshot")?.call0()?;
            let available = memory
                .call_method1("get", ("available", u128::from(u64::MAX) + 1))?
                .extract::<u128>()?;
            if available < 2 * 1024 * 1024 * 1024 {
                policy(host)?.borrow_mut(py).admission = Admission::RamPressure;
                return Ok(executor.unbind());
            }
        }
        policy(host)?.borrow_mut(py).admission = Admission::Open;
        let queue = host.getattr("queue")?;
        let kw = PyDict::new(py);
        kw.set_item("timeout", 0.1)?;
        let item = queue.call_method("get", (), Some(&kw))?;
        if item.is_none() {
            host.call_method0("housekeeping")?;
            return Ok(executor.unbind());
        }
        let prompt = item.get_item(0)?;
        let item_id = item.get_item(1)?;
        let prompt_id = prompt.get_item(1)?;
        let context = slots.call_method1("job", (&prompt_id, &executor))?;
        let job = context.call_method0("__enter__")?;
        let result = execute_job(
            host,
            &executor,
            source,
            prompt_queue,
            &prompt,
            &item_id,
            &job,
        );
        let context_result = match &result {
            Ok(_) => context.call_method1("__exit__", (py.None(), py.None(), py.None())),
            Err(err) => context.call_method1(
                "__exit__",
                (err.get_type(py), err.value(py), err.traceback(py)),
            ),
        };
        context_result?;
        result?;
        host.call_method0("housekeeping")?;
        Ok(executor.unbind())
    })
}
fn execute_job(
    host: &Bound<'_, PyAny>,
    executor: &Bound<'_, PyAny>,
    source: &Bound<'_, PyModule>,
    prompt_queue: &Bound<'_, PyAny>,
    prompt: &Bound<'_, PyAny>,
    item_id: &Bound<'_, PyAny>,
    job: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = host.py();
    let started = now(source, "perf_counter")?;
    let prompt_id = prompt.get_item(1)?;
    let id: String = prompt_id.extract()?;
    host.getattr("assets")?
        .call_method0("pause_background_scan")?;
    let server = host.getattr("server")?;
    server.setattr("last_prompt_id", &prompt_id)?;
    server.setattr(
        "workflow_metadata",
        prompt
            .get_item(3)?
            .call_method1("get", ("workflow_metadata", PyDict::new(py)))?,
    )?;
    let extra = prompt.get_item(3)?.cast_into::<PyDict>()?.copy()?;
    extra.update(prompt.get_item(5)?.cast::<pyo3::types::PyMapping>()?)?;
    job.setattr("extra", &extra)?;
    job.setattr("outputs", prompt.get_item(4)?)?;
    let execution = (|| {
        let cancelled = with_context(&host.getattr("queue")?.getattr("mutex")?, || {
            let state = policy(host)?;
            let mut state = state.borrow_mut(py);
            let pending = state.pending_cancel.remove(&id) || state.stopping;
            drop(state);
            Ok(pending || job.getattr("cancel")?.call_method0("is_set")?.is_truthy()?)
        })?;
        if cancelled {
            executor.setattr("status_messages", PyList::empty(py))?;
            let data = PyDict::new(py);
            data.set_item("prompt_id", &prompt_id)?;
            data.set_item("node_id", py.None())?;
            data.set_item("executed", PyList::empty(py))?;
            let kw = PyDict::new(py);
            kw.set_item("broadcast", true)?;
            executor.call_method("add_message", ("execution_interrupted", data), Some(&kw))?;
            failed(executor)?;
        } else {
            executor.call_method1(
                "execute",
                (prompt.get_item(2)?, &prompt_id, extra, prompt.get_item(4)?),
            )?;
        }
        Ok::<(), PyErr>(())
    })();
    let escaped = match execution {
        Err(error) if error.is_instance_of::<PyException>(py) => {
            log(source, "Pipeline execution failed", &error)?;
            failed(executor)?;
            None
        }
        Err(error) => Some(error),
        Ok(_) => None,
    };
    // Completion mirrors the host finally block, including uncaught BaseException paths.
    let kw = PyDict::new(py);
    let state = PyDict::new(py);
    let success = executor.getattr("success")?;
    state.set_item(
        "status_str",
        if success.is_truthy()? {
            "success"
        } else {
            "error"
        },
    )?;
    state.set_item("completed", &success)?;
    state.set_item("messages", executor.getattr("status_messages")?)?;
    kw.set_item(
        "status",
        prompt_queue
            .getattr("ExecutionStatus")?
            .call((), Some(&state))?,
    )?;
    kw.set_item("process_item", wrap_pyfunction!(host_history_item, py)?)?;
    host.getattr("queue")?.call_method(
        "task_done",
        (item_id, executor.getattr("history_result")?),
        Some(&kw),
    )?;
    let done = PyDict::new(py);
    done.set_item("node", py.None())?;
    done.set_item("prompt_id", prompt_id)?;
    server.call_method1(
        "send_sync",
        ("executing", done, server.getattr("client_id")?),
    )?;
    server.setattr("workflow_metadata", PyDict::new(py))?;
    source.getattr("logging")?.call_method1(
        "info",
        (
            "Prompt executed in %.2f seconds",
            now(source, "perf_counter")? - started,
        ),
    )?;
    match escaped {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
fn failed(executor: &Bound<'_, PyAny>) -> PyResult<()> {
    executor.setattr("success", false)?;
    let result = PyDict::new(executor.py());
    result.set_item("outputs", PyDict::new(executor.py()))?;
    executor.setattr("history_result", result)
}
#[pyfunction]
fn host_remote(
    host: &Bound<'_, PyAny>,
    operation: &Bound<'_, PyAny>,
    models: &Bound<'_, PyAny>,
    cancellation_scope: &Bound<'_, PyAny>,
    cancelled_class: &Bound<'_, PyAny>,
    get_context: &Bound<'_, PyAny>,
    source: &Bound<'_, PyModule>,
) -> PyResult<Py<PyAny>> {
    let py = host.py();
    let slots = host.getattr("slots")?;
    let job = slots
        .getattr("thread")?
        .getattr("job")
        .unwrap_or_else(|_| py.None().into_bound(py));
    if job.is_none() {
        return Ok(operation.call0()?.unbind());
    }
    models.call_method0("throw_exception_if_processing_interrupted")?;
    let context = get_context.call0()?;
    let operation = operation.clone().unbind();
    let job_call = job.clone().unbind();
    let scope = cancellation_scope.clone().unbind();
    let call = PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<Py<PyAny>> {
            let py = args.py();
            let kw = PyDict::new(py);
            kw.set_item("caller_interrupt", false)?;
            let scope = scope.bind(py).call(
                (job_call.bind(py).getattr("cancel")?.getattr("is_set")?,),
                Some(&kw),
            )?;
            with_context(&scope, || operation.call0(py))
        },
    )?;
    let host_complete = host.clone().unbind();
    let job_complete = job.clone().unbind();
    let context = context.unbind();
    let source = source.clone().unbind();
    let completed = PyCFunction::new_closure(
        py,
        None,
        None,
        move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<()> {
            let py = args.py();
            let host = host_complete.bind(py);
            let job = job_complete.bind(py);
            let context = context.bind(py);
            let previews = host.getattr("previews")?;
            if previews.is_truthy()? && !context.is_none() {
                let decoder = context.getattr("node_id")?;
                if !decoder.is_none() {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("list_index", context.getattr("list_index")?)?;
                    if let Err(error) = previews.call_method(
                        "publish",
                        (
                            job,
                            args.get_item(1)?
                                .get_item("progress")?
                                .getattr("dynprompt")?,
                            decoder,
                            args.get_item(0)?,
                            job.getattr("extra")?,
                            job.getattr("outputs")?,
                        ),
                        Some(&kwargs),
                    ) {
                        if error.is_instance_of::<PyException>(py) {
                            log(
                                source.bind(py),
                                "Early preview unavailable; native output retained",
                                &error,
                            )?;
                        } else {
                            return Err(error);
                        }
                    }
                }
            }
            Ok(())
        },
    )?;
    let kw = PyDict::new(py);
    kw.set_item("completed", completed)?;
    match slots.call_method("remote", (call,), Some(&kw)) {
        Ok(result) => Ok(result.unbind()),
        Err(error) if error.value(py).is_instance(cancelled_class)? => {
            let translated =
                PyErr::from_value(models.getattr("InterruptProcessingException")?.call0()?);
            translated.set_cause(py, Some(error));
            Err(translated)
        }
        Err(error) => Err(error),
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HostPolicy>()?;
    m.add_function(wrap_pyfunction!(host_begin_shutdown, m)?)?;
    m.add_class::<PressureRelief>()?;
    m.add_function(wrap_pyfunction!(host_status, m)?)?;
    m.add_function(wrap_pyfunction!(host_share_models, m)?)?;
    m.add_function(wrap_pyfunction!(host_executor, m)?)?;
    m.add_function(wrap_pyfunction!(host_cancel, m)?)?;
    m.add_function(wrap_pyfunction!(host_interrupt, m)?)?;
    m.add_function(wrap_pyfunction!(host_housekeeping, m)?)?;
    m.add_function(wrap_pyfunction!(host_history_item, m)?)?;
    m.add_function(wrap_pyfunction!(host_worker_tick, m)?)?;
    m.add_function(wrap_pyfunction!(host_remote, m)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_precedence_and_memory_bounds() {
        assert_eq!(cache_selection(true, 10, true), CacheSelection::Classic);
        assert_eq!(cache_selection(false, 1, true), CacheSelection::Lru);
        assert_eq!(cache_selection(false, 0, true), CacheSelection::None);
        assert_eq!(cache_budgets(1024.0, &[]), (2.0, 1.0));
        assert_eq!(cache_budgets(1024.0 * 1000.0, &[]), (10.0, 128.0));
        assert_eq!(cache_budgets(1024.0, &[3.0, 9.0]), (3.0, 9.0));
    }
}

#[pyclass]
struct PressureRelief {
    clear: Py<PyAny>,
    source: Py<PyModule>,
    last: f64,
}
#[pymethods]
impl PressureRelief {
    #[new]
    fn new(clear: Py<PyAny>, source: Py<PyModule>) -> Self {
        Self {
            clear,
            source,
            last: 0.0,
        }
    }
    fn tick(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let source = self.source.bind(py);
        let state = source.getattr("snapshot")?.call0()?;
        let available = state
            .call_method1("get", ("available",))?
            .extract::<Option<u64>>()?;
        if super::memory_pressure(available, now(source, "monotonic")?, self.last) {
            self.clear.call0(py)?;
            source.getattr("gc")?.call_method0("collect")?;
            self.last = now(source, "monotonic")?;
        }
        Ok(state.unbind())
    }
}

#[pyfunction]
fn host_begin_shutdown(host: &Bound<'_, PyAny>) -> PyResult<()> {
    policy(host)?.borrow_mut(host.py()).stopping = true;
    let slots = host.getattr("slots")?;
    for job in slots.getattr("jobs")?.call_method0("values")?.try_iter()? {
        job?.getattr("cancel")?.call_method0("set")?;
    }
    if !slots.getattr("active")?.is_none() {
        host.getattr("entry")?
            .getattr("nodes")?
            .call_method0("interrupt_processing")?;
    }
    Ok(())
}

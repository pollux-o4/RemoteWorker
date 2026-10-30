//! Managed host invocation and foreground/decode ordering.
use super::actions::with_context;
use pyo3::{
    prelude::*,
    types::{PyCFunction, PyDict, PyTuple},
};
#[pyclass]
struct ForegroundLease {
    source: Py<PyModule>,
    url: String,
    decode: bool,
    check_wait: Py<PyAny>,
    lease: Option<super::GateLease>,
}
#[pymethods]
impl ForegroundLease {
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.lease.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Foreground lease already entered",
            ));
        }
        loop {
            self.check_wait.call0(py)?;
            let source = self.source.bind(py);
            let coordinator = source.getattr("coordinator")?;
            if self.decode && coordinator.is_truthy()? {
                coordinator.call_method1("ensure_next_before_decode", (&self.url,))?;
            }
            let gate = super::gate(&self.url);
            let mut lease = gate.enter(false, Some(self.check_wait.clone_ref(py)));
            lease.__enter__(py)?;
            if self.decode
                && coordinator.is_truthy()?
                && coordinator
                    .call_method1(
                        "next_clip_missing",
                        (source.getattr("current_prompt")?.call0()?, &self.url),
                    )?
                    .is_truthy()?
            {
                lease.release();
                continue;
            }
            self.lease = Some(lease);
            return Ok(());
        }
    }
    fn __exit__(
        &mut self,
        _kind: &Bound<'_, PyAny>,
        _error: &Bound<'_, PyAny>,
        _trace: &Bound<'_, PyAny>,
    ) -> bool {
        self.lease = None;
        false
    }
}
#[pyfunction]
fn prep_foreground_lease(
    source: Py<PyModule>,
    url: String,
    decode: bool,
    check_wait: Py<PyAny>,
) -> ForegroundLease {
    ForegroundLease {
        source,
        url,
        decode,
        check_wait,
        lease: None,
    }
}
#[pyfunction]
#[pyo3(signature=(source,function,args,kwargs,values,kind=None))]
fn prep_managed_call(
    source: &Bound<'_, PyModule>,
    function: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kwargs: &Bound<'_, PyDict>,
    values: &Bound<'_, PyDict>,
    kind: Option<String>,
) -> PyResult<Py<PyAny>> {
    let py = source.py();
    let url = values.get_item("server_url")?;
    if url.as_ref().is_none_or(|u| !u.is_truthy().unwrap_or(false))
        || source.getattr("is_speculative")?.call0()?.is_truthy()?
    {
        return Ok(function.call(args, Some(kwargs))?.unbind());
    }
    let url = url.unwrap();
    let coordinator = source.getattr("coordinator")?;
    let host = source.getattr("pipeline")?.getattr("runtime")?;
    if kind.as_deref() == Some("decoder") && host.is_truthy()? {
        let job = host
            .getattr("slots")?
            .getattr("thread")?
            .getattr("job")
            .unwrap_or_else(|_| py.None().into_bound(py));
        if !job.is_none() {
            let prompt = source.getattr("current_prompt")?.call0()?;
            let following = if coordinator.is_truthy()? {
                source
                    .getattr("head")?
                    .call1((coordinator.call_method0("queue")?.get_item(1)?,))?
            } else {
                py.None().into_bound(py)
            };
            let source = source.clone().unbind();
            let coordinator = coordinator.unbind();
            let function = function.clone().unbind();
            let args = args.clone().unbind();
            let kwargs = kwargs.clone().unbind();
            let url = url.unbind();
            let prompt = prompt.unbind();
            let following = following.unbind();
            let decode = PyCFunction::new_closure(
                py,
                None,
                None,
                move |call: &Bound<'_, PyTuple>,
                      _: Option<&Bound<'_, PyDict>>|
                      -> PyResult<Py<PyAny>> {
                    let py = call.py();
                    let source = source.bind(py);
                    let coordinator = coordinator.bind(py);
                    if coordinator.is_truthy()? {
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("current_id", prompt.bind(py))?;
                        kwargs.set_item("following", following.bind(py))?;
                        coordinator.call_method(
                            "ensure_next_before_decode",
                            (url.bind(py),),
                            Some(&kwargs),
                        )?;
                    }
                    let gate = super::gate(&url.bind(py).extract::<String>()?);
                    let lease = Py::new(
                        py,
                        gate.enter(
                            false,
                            Some(source.getattr("_raise_if_caller_interrupted")?.unbind()),
                        ),
                    )?;
                    with_context(lease.bind(py).as_any(), || {
                        Ok(function
                            .bind(py)
                            .call(args.bind(py), Some(kwargs.bind(py)))?
                            .unbind())
                    })
                },
            )?;
            return Ok(host.call_method1("remote", (decode,))?.unbind());
        }
    }
    let source_check = source.clone().unbind();
    let values_check = values.clone().unbind();
    let kind_check = kind.clone();
    let url_check = url.clone().unbind();
    let check_prepared = PyCFunction::new_closure(
        py,
        None,
        None,
        move |call: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<()> {
            let py = call.py();
            let source = source_check.bind(py);
            source.getattr("_raise_if_caller_interrupted")?.call0()?;
            let coordinator = source.getattr("coordinator")?;
            if coordinator.is_truthy()? && kind_check.as_deref() == Some("encoder") {
                let handoffs = coordinator.getattr("handoffs")?;
                if handoffs.is_truthy()? {
                    let prompt = source.getattr("current_prompt")?.call0()?;
                    let node = values_check
                        .bind(py)
                        .get_item("unique_id")?
                        .unwrap_or_else(|| py.None().into_bound(py))
                        .str()?;
                    let key = PyTuple::new(py, [prompt, node.into_any()])?;
                    let stamp = super::handoff_signature(values_check.bind(py))?;
                    let blob = handoffs.call_method1("take", (&key, stamp))?;
                    if !blob.is_none() {
                        let fields = PyDict::new(py);
                        fields.set_item("prompt_id", key.get_item(0)?)?;
                        fields.set_item("node_id", key.get_item(1)?)?;
                        fields.set_item("peer", url_check.bind(py))?;
                        coordinator.call_method(
                            "record",
                            ("conditioning_consumed",),
                            Some(&fields),
                        )?;
                        return Err(PyErr::from_value(
                            source
                                .getattr("PreparedResult")?
                                .call1((super::handoff_unpack(&blob)?,))?,
                        ));
                    }
                }
            }
            Ok(())
        },
    )?;
    let check_wait = if kind.as_deref() == Some("encoder") {
        check_prepared.into_any()
    } else {
        py.None().into_bound(py)
    };
    let foreground = source.getattr("foreground_gate")?.call1((
        &url,
        kind.as_deref() == Some("decoder"),
        check_wait,
    ))?;
    let result = with_context(&foreground, || {
        if coordinator.is_truthy()? && kind.as_deref() == Some("encoder") {
            coordinator.call_method1(
                "encoder_started",
                (
                    source.getattr("current_prompt")?.call0()?,
                    values
                        .get_item("unique_id")?
                        .unwrap_or_else(|| py.None().into_bound(py)),
                ),
            )?;
        }
        let started: f64 = source
            .getattr("time")?
            .call_method0("monotonic")?
            .extract()?;
        let result = function.call(args, Some(kwargs))?;
        if coordinator.is_truthy()? && kind.as_deref() == Some("encoder") {
            let scheduler = coordinator.getattr("scheduler")?;
            if scheduler.is_truthy()? {
                let now: f64 = source
                    .getattr("time")?
                    .call_method0("monotonic")?
                    .extract()?;
                scheduler.call_method1("observe_encoder", (now - started,))?;
            }
        }
        Ok(result.unbind())
    });
    match result {
        Err(err)
            if err
                .value(py)
                .is_instance(&source.getattr("PreparedResult")?)? =>
        {
            Ok(err.value(py).getattr("result")?.unbind())
        }
        other => other,
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prep_bind_values, m)?)?;
    m.add_function(wrap_pyfunction!(prep_peer_available, m)?)?;
    m.add_function(wrap_pyfunction!(prep_require_idle_peer, m)?)?;
    m.add_function(wrap_pyfunction!(prep_register_targets, m)?)?;
    m.add_function(wrap_pyfunction!(prep_resolve_inputs, m)?)?;
    m.add_class::<ForegroundLease>()?;
    m.add_function(wrap_pyfunction!(prep_foreground_lease, m)?)?;
    m.add_function(wrap_pyfunction!(prep_managed_call, m)?)?;
    Ok(())
}

#[pyfunction]
fn prep_bind_values(
    function: &Bound<'_, PyAny>,
    values: &Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    let py = values.py();
    let bound = py
        .import("inspect")?
        .call_method1("signature", (function,))?
        .call_method("bind", (py.None(),), Some(values))?;
    bound.call_method0("apply_defaults")?;
    let result = bound.getattr("arguments")?.cast_into::<PyDict>()?.copy()?;
    if result.contains("self")? {
        result.del_item("self")?;
    }
    Ok(result.unbind())
}
#[pyfunction]
fn prep_peer_available(queue: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(!queue.call_method1("get", ("queue_running",))?.is_truthy()?
        && !queue.call_method1("get", ("queue_pending",))?.is_truthy()?)
}
#[pyfunction]
fn prep_require_idle_peer(queue: &Bound<'_, PyAny>) -> PyResult<()> {
    if !prep_peer_available(queue)? {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("peer_busy"));
    }
    Ok(())
}
#[pyfunction]
fn prep_register_targets(
    capabilities: &Bound<'_, PyDict>,
    managed: &Bound<'_, PyAny>,
) -> PyResult<Py<PyDict>> {
    let py = capabilities.py();
    let result = PyDict::new(py);
    for (name, value) in capabilities.iter() {
        let cls = value.get_item(0)?;
        let spec = value.get_item(1)?;
        let executor = spec.getattr("executor")?;
        let target = if executor.is_truthy()? { executor } else { cls };
        let role = spec.getattr("role")?;
        target.setattr("run", managed.call1((target.getattr("run")?, &role))?)?;
        if role.extract::<String>()? == "encoder" {
            result.set_item(name, target)?;
        }
    }
    Ok(result.unbind())
}
#[pyfunction]
fn prep_resolve_inputs(
    graph: Py<PyDict>,
    classes: Py<PyDict>,
    node_id: &str,
    decoders: &Bound<'_, PyAny>,
    check_cancelled: Py<PyAny>,
) -> PyResult<Py<PyDict>> {
    let py = decoders.py();
    let node = graph
        .bind(py)
        .get_item(node_id)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(node_id.to_owned()))?;
    let class = node.get_item("class_type")?;
    let exclude = if decoders.contains(class)? {
        PyTuple::new(py, ["samples", "warmup", "latent"])?
    } else {
        PyTuple::empty(py)
    };
    super::Resolver::new(py, graph, classes, check_cancelled).resolve_inputs(
        py,
        node_id,
        Some(exclude.as_any()),
    )
}

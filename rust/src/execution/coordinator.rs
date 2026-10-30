//! Bounded queue reconciliation and lookahead; callbacks are host integration only.
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PySet, PyTuple},
};
fn kwargs<'py>(
    py: Python<'py>,
    pairs: &[(&str, Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in pairs {
        dict.set_item(k, v)?;
    }
    Ok(dict)
}
fn module<'py>(owner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    owner
        .py()
        .import(
            owner
                .get_type()
                .getattr("__module__")?
                .extract::<String>()?
                .as_str(),
        )
        .map(Bound::into_any)
}
fn plan<'py>(owner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    module(owner)?.getattr("plan")
}
fn queue<'py>(owner: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let value = owner.call_method0("queue")?;
    Ok((value.get_item(0)?, value.get_item(1)?))
}
fn handled(py: Python<'_>, err: &PyErr) -> bool {
    err.is_instance_of::<PyOSError>(py)
        || err.is_instance_of::<PyRuntimeError>(py)
        || err.is_instance_of::<PyValueError>(py)
}
fn cancelled(owner: &Bound<'_, PyAny>, err: &PyErr) -> PyResult<bool> {
    err.value(owner.py())
        .is_instance(&module(owner)?.getattr("RequestCancelled")?)
}
fn record(owner: &Bound<'_, PyAny>, state: &str, fields: &Bound<'_, PyDict>) -> PyResult<()> {
    owner.call_method("record", (state,), Some(fields))?;
    Ok(())
}
fn ids<'py>(
    py: Python<'py>,
    running: &Bound<'py, PyAny>,
    pending: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PySet>> {
    let ids = PySet::empty(py)?;
    for values in [running, pending] {
        for item in values.try_iter()? {
            ids.add(item?.get_item(1)?)?;
        }
    }
    Ok(ids)
}
#[pyfunction]
fn prep_queue(owner: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let queue = owner.getattr("server")?.getattr("prompt_queue")?;
    let method = queue
        .getattr("get_current_queue_volatile")
        .or_else(|_| queue.getattr("get_current_queue"))?;
    let result = method.call0()?;
    let running = result.get_item(0)?;
    let pending = result.get_item(1)?;
    let active = module(owner)?
        .getattr("pipeline")?
        .call_method0("status")?
        .call_method1("get", ("local_prompt_id",))?;
    if active.is_truthy()? {
        let mut first = Vec::new();
        let mut rest = Vec::new();
        for item in running.try_iter()? {
            let item = item?;
            if item.get_item(1)?.eq(&active)? {
                first.push(item);
            } else {
                rest.push(item);
            }
        }
        first.extend(rest);
        return Ok(
            PyTuple::new(py, [PyList::new(py, first)?.into_any(), pending])?
                .into_any()
                .unbind(),
        );
    }
    Ok(result.unbind())
}
#[pyfunction]
fn prep_record(owner: &Bound<'_, PyAny>, state: &str, fields: &Bound<'_, PyDict>) -> PyResult<()> {
    let py = owner.py();
    let value = fields.copy()?;
    value.set_item("state", state)?;
    let active = owner.getattr("active")?;
    if active.is_truthy()? {
        let comparable = active.cast::<PyDict>()?.copy()?;
        comparable.del_item("at").ok();
        if comparable.eq(&value)? {
            return Ok(());
        }
    }
    value.set_item("at", py.import("time")?.call_method0("time")?)?;
    owner.setattr("active", &value)?;
    owner
        .getattr("events")?
        .call_method1("append", (value.copy()?,))?;
    Ok(())
}
#[pyfunction]
fn prep_status(owner: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let py = owner.py();
    let (running, pending) = queue(owner)?;
    let ids = ids(py, &running, &pending)?;
    let module = module(owner)?;
    let pipeline = module.getattr("pipeline")?;
    let runtime = pipeline.getattr("runtime")?.is_truthy()?;
    let result = PyDict::new(py);
    result.set_item("enabled", true)?;
    result.set_item("pipeline", pipeline.call_method0("status")?)?;
    let active = owner.getattr("active")?;
    result.set_item(
        "active",
        if active.is_truthy()? && ids.contains(active.call_method1("get", ("prompt_id",))?)? {
            active
        } else {
            py.None().into_bound(py)
        },
    )?;
    result.set_item(
        "events",
        py.import("builtins")?
            .getattr("list")?
            .call1((owner.getattr("events")?,))?,
    )?;
    let counts = PyDict::new(py);
    counts.set_item("running", running.len()?)?;
    counts.set_item("pending", pending.len()?)?;
    result.set_item("queue", counts)?;
    result.set_item("lookahead", 1)?;
    result.set_item(
        "scope",
        if owner.getattr("resolve")?.is_truthy()? {
            "cpu_ancestors"
        } else {
            "literal_inputs"
        },
    )?;
    let handoffs = owner.getattr("handoffs")?;
    result.set_item(
        "handoff_bytes",
        if handoffs.is_truthy()? {
            handoffs.getattr("bytes_used")?
        } else {
            0_u64.into_pyobject(py)?.into_any()
        },
    )?;
    result.set_item(
        "vae_policy",
        if runtime {
            "real_decode_pipeline"
        } else {
            "first_decoder"
        },
    )?;
    result.set_item("memory", module.getattr("snapshot")?.call0()?)?;
    let scheduler = owner.getattr("scheduler")?;
    if scheduler.is_truthy()? {
        let status = scheduler.call_method0("status")?.cast_into::<PyDict>()?;
        if runtime {
            status.set_item("policy", "clip_first_pipeline")?;
        }
        status.set_item("driver", "events")?;
        status.set_item("reconcile_seconds", 5)?;
        status.set_item("ticks", owner.getattr("ticks")?)?;
        result.set_item("scheduler", status)?;
    } else {
        result.set_item("scheduler", py.None())?;
    }
    Ok(result.unbind())
}
#[pyfunction]
fn prep_eligible(owner: &Bound<'_, PyAny>, prompt_id: &Bound<'_, PyAny>) -> PyResult<bool> {
    let (running, _) = queue(owner)?;
    if running.len()? == 0 {
        return Ok(false);
    }
    let current = running.get_item(0)?;
    Ok(current.get_item(1)?.eq(prompt_id)?
        && owner
            .getattr("sampling")?
            .call1((prompt_id, current.get_item(2)?))?
            .is_truthy()?)
}
#[pyfunction]
fn prep_job_present(
    owner: &Bound<'_, PyAny>,
    prompt_id: &Bound<'_, PyAny>,
    node_id: &Bound<'_, PyAny>,
    expected: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let (running, pending) = queue(owner)?;
    for jobs in [running, pending] {
        for item in jobs.try_iter()? {
            let item = item?;
            if item.get_item(1)?.eq(prompt_id)? {
                let node = item.get_item(2)?.call_method1("get", (node_id,))?;
                return Ok(!node.is_none()
                    && plan(owner)?
                        .call_method1("fingerprint", (node,))?
                        .eq(expected)?);
            }
        }
    }
    Ok(false)
}
#[pyfunction]
fn prep_prompt_present(owner: &Bound<'_, PyAny>, prompt_id: &Bound<'_, PyAny>) -> PyResult<bool> {
    let (running, pending) = queue(owner)?;
    for jobs in [running, pending] {
        for item in jobs.try_iter()? {
            if item?.get_item(1)?.eq(prompt_id)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
#[pyfunction]
fn prep_decoder(owner: &Bound<'_, PyAny>, current: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    let plan = plan(owner)?;
    let graph = plan.call_method1("job_graph", (current,))?;
    let resolve = owner.getattr("resolve")?;
    if !resolve.is_truthy()? {
        return Ok(plan.call_method1("first_decoder", (graph,))?.unbind());
    }
    let graph = graph.cast_into::<PyDict>()?;
    let adjusted = graph.copy()?;
    let decoders = plan.call_method0("decoder_types")?;
    for (key, node) in graph.iter() {
        if !decoders.contains(node.call_method1("get", ("class_type",))?)? {
            continue;
        }
        let inputs = node.call_method1("get", ("inputs", PyDict::new(py)))?;
        if !inputs.call_method1("get", ("warmup",))?.is_none() {
            continue;
        }
        let values = match resolve.call1((&graph, &key)) {
            Ok(values) => {
                let values = values.cast_into::<PyDict>()?;
                if let Some(override_name) = values
                    .get_item("vae_name_override")?
                    .filter(|v| !v.is_none())
                {
                    values.set_item("vae_name", override_name)?;
                }
                values
            }
            Err(err) if err.is_instance_of::<PyValueError>(py) => PyDict::new(py),
            Err(err) => return Err(err),
        };
        let updated = node.cast::<PyDict>()?.copy()?;
        updated.set_item("inputs", values)?;
        adjusted.set_item(key, updated)?;
    }
    Ok(plan.call_method1("first_decoder", (adjusted,))?.unbind())
}
#[pyfunction]
fn prep_tick(owner: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = owner.py();
    let count: u64 = owner.getattr("ticks")?.extract()?;
    owner.setattr("ticks", count + 1)?;
    let relief = owner.getattr("relief")?;
    if relief.is_truthy()? {
        relief.call_method0("tick")?;
    }
    let (running, pending) = queue(owner)?;
    let ids = ids(py, &running, &pending)?;
    for name in ["handoffs", "scheduler"] {
        let component = owner.getattr(name)?;
        if component.is_truthy()? {
            component.call_method1("retain", (&ids,))?;
        }
    }
    if running.len()? == 0 {
        return Ok(());
    }
    let current = running.get_item(0)?;
    if !owner
        .call_method1("eligible", (current.get_item(1)?,))?
        .is_truthy()?
    {
        return Ok(());
    }
    owner.call_method1("prepare_next", (&current, &pending))?;
    let module = module(owner)?;
    if !module.getattr("pipeline")?.getattr("runtime")?.is_none() {
        return Ok(());
    }
    let decoder = owner.call_method1("decoder", (&current,))?;
    let scheduler = owner.getattr("scheduler")?;
    if scheduler.is_truthy()? {
        let following = if pending.len()? > 0 {
            plan(owner)?
                .call_method1("head", (&pending,))?
                .get_item(1)?
        } else {
            py.None().into_bound(py)
        };
        let needs_clip = decoder.is_truthy()?
            && owner
                .call_method1(
                    "next_clip_missing",
                    (current.get_item(1)?, decoder.get_item(1)?),
                )?
                .is_truthy()?;
        let kw = PyDict::new(py);
        kw.set_item("needs_clip", needs_clip)?;
        kw.set_item("decoder", &decoder)?;
        let decision =
            scheduler.call_method("decide", (current.get_item(1)?, following), Some(&kw))?;
        if decision.getattr("action")?.extract::<String>()? != "warmup" {
            return Ok(());
        }
    }
    module
        .getattr("prepare_vae")?
        .call1((owner, current, decoder))?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature=(owner,current,pending=None,*,allow_after_sampling=false,peer=None))]
fn prep_prepare_next(
    owner: &Bound<'_, PyAny>,
    current: &Bound<'_, PyAny>,
    pending: Option<&Bound<'_, PyAny>>,
    allow_after_sampling: bool,
    peer: Option<&str>,
) -> PyResult<()> {
    let py = owner.py();
    let pending = match pending {
        Some(p) => p.clone(),
        None => queue(owner)?.1,
    };
    let done = owner.getattr("done")?;
    let encoders = plan(owner)?.call_method1("pending_encoders", (&pending, &done))?;
    for row in encoders.try_iter()? {
        let row = row?;
        let following = row.get_item(0)?;
        let graph = row.get_item(1)?;
        let node_id = row.get_item(2)?;
        let node = row.get_item(3)?;
        let mut values = row.get_item(4)?;
        let token = row.get_item(5)?;
        let resolve = owner.getattr("resolve")?;
        if resolve.is_truthy()? {
            match resolve.call1((&graph, &node_id)) {
                Ok(resolved) => values = resolved,
                Err(err) if err.is_instance_of::<PyValueError>(py) => {
                    done.call_method1("append", (&token,))?;
                    record(
                        owner,
                        "skipped",
                        &kwargs(
                            py,
                            &[
                                ("prompt_id", following.get_item(1)?),
                                ("node_id", node_id.clone()),
                                ("reason", err.value(py).str()?.into_any()),
                            ],
                        )?,
                    )?;
                    continue;
                }
                Err(err) => return Err(err),
            }
        }
        if values.is_none() {
            done.call_method1("append", (&token,))?;
            record(
                owner,
                "skipped",
                &kwargs(
                    py,
                    &[
                        ("prompt_id", following.get_item(1)?),
                        ("node_id", node_id.clone()),
                        (
                            "reason",
                            "dynamic_inputs_or_refresh".into_pyobject(py)?.into_any(),
                        ),
                    ],
                )?,
            )?;
            continue;
        }
        let url: String = values.call_method1("get", ("server_url", ""))?.extract()?;
        if peer.is_some_and(|p| p.trim_end_matches('/') != url.trim_end_matches('/')) {
            continue;
        }
        if url.is_empty() || owner.call_method1("local_peer", (&url,))?.is_truthy()? {
            done.call_method1("append", (&token,))?;
            record(
                owner,
                "skipped",
                &kwargs(
                    py,
                    &[
                        (
                            "reason",
                            "local_or_missing_peer".into_pyobject(py)?.into_any(),
                        ),
                        ("node_id", node_id.clone()),
                    ],
                )?,
            )?;
            continue;
        }
        if !allow_after_sampling && !owner.getattr("available")?.call1((&url,))?.is_truthy()? {
            record(
                owner,
                "deferred",
                &kwargs(
                    py,
                    &[
                        ("prompt_id", following.get_item(1)?),
                        ("reason", "peer_busy".into_pyobject(py)?.into_any()),
                    ],
                )?,
            )?;
            return Ok(());
        }
        let kw = PyDict::new(py);
        kw.set_item("allow_after_sampling", allow_after_sampling)?;
        if let Err(err) = module(owner)?.getattr("prefetch")?.call(
            (
                owner, current, following, graph, node_id, node, values, &token,
            ),
            Some(&kw),
        ) {
            if handled(py, &err) {
                done.call_method1("append", (token,))?;
            }
            return Err(err);
        }
    }
    Ok(())
}
#[pyfunction]
fn prep_encoder_started(
    owner: &Bound<'_, PyAny>,
    prompt_id: &Bound<'_, PyAny>,
    node_id: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let (running, _) = queue(owner)?;
    for item in running.try_iter()? {
        let item = item?;
        if item.get_item(1)?.eq(prompt_id)? {
            let node_id = node_id.str()?;
            let node = item.get_item(2)?.call_method1("get", (&node_id,))?;
            if !node.is_none() {
                let token = PyTuple::new(
                    owner.py(),
                    [
                        prompt_id.clone(),
                        node_id.into_any(),
                        plan(owner)?.call_method1("fingerprint", (node,))?,
                    ],
                )?;
                let done = owner.getattr("done")?;
                if !done.contains(&token)? {
                    done.call_method1("append", (token,))?;
                }
            }
            return Ok(());
        }
    }
    Ok(())
}
#[pyfunction]
fn prep_next_clip_missing(
    owner: &Bound<'_, PyAny>,
    current_id: &Bound<'_, PyAny>,
    url: &str,
) -> PyResult<bool> {
    let (running, pending) = queue(owner)?;
    if running.len()? == 0
        || !running.get_item(0)?.get_item(1)?.eq(current_id)?
        || pending.len()? == 0
    {
        return Ok(false);
    }
    for row in plan(owner)?
        .call_method1("pending_encoders", (pending, owner.getattr("done")?))?
        .try_iter()?
    {
        let node = row?.get_item(3)?;
        let peer = node
            .call_method1("get", ("inputs", PyDict::new(owner.py())))?
            .call_method1("get", ("server_url",))?;
        if peer
            .extract::<String>()
            .is_ok_and(|peer| peer.trim_end_matches('/') == url.trim_end_matches('/'))
        {
            return Ok(true);
        }
    }
    Ok(false)
}
#[pyfunction]
#[pyo3(signature=(owner,url,*,current_id=None,following=None))]
fn prep_ensure_next_before_decode(
    owner: &Bound<'_, PyAny>,
    url: &str,
    current_id: Option<&Bound<'_, PyAny>>,
    following: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let py = owner.py();
    let mut following = following.cloned();
    loop {
        let (running, pending) = queue(owner)?;
        if running.len()? == 0 || (pending.len()? == 0 && following.is_none()) {
            return Ok(());
        }
        let current = if let Some(id) = current_id {
            let mut selected = None;
            for item in running.try_iter()? {
                let item = item?;
                if item.get_item(1)?.eq(id)? {
                    selected = Some(item);
                    break;
                }
            }
            let Some(selected) = selected else {
                return Ok(());
            };
            selected
        } else {
            running.get_item(0)?
        };
        let candidates = if current_id.is_some() {
            PyList::new(
                py,
                [match &following {
                    Some(following) => following.clone(),
                    None => plan(owner)?.call_method1("head", (&pending,))?,
                }],
            )?
            .into_any()
        } else {
            pending
        };
        let kw = PyDict::new(py);
        kw.set_item("allow_after_sampling", true)?;
        kw.set_item("peer", url)?;
        match owner.call_method("prepare_next", (&current, candidates), Some(&kw)) {
            Ok(_) => {
                if current_id.is_some()
                    || !owner
                        .call_method1("next_clip_missing", (current.get_item(1)?, url))?
                        .is_truthy()?
                {
                    return Ok(());
                }
            }
            Err(err) if cancelled(owner, &err)? => {
                following = None;
                if !owner
                    .call_method1("prompt_present", (current.get_item(1)?,))?
                    .is_truthy()?
                {
                    return Ok(());
                }
            }
            Err(err) if handled(py, &err) => {
                let active = owner.getattr("active")?;
                let prompt = if active.is_truthy()? {
                    active.call_method1("get", ("prompt_id",))?
                } else {
                    py.None().into_bound(py)
                };
                let reason: String = err.value(py).str()?.extract()?;
                record(
                    owner,
                    "error",
                    &kwargs(
                        py,
                        &[
                            ("prompt_id", prompt),
                            (
                                "reason",
                                reason
                                    .chars()
                                    .take(200)
                                    .collect::<String>()
                                    .into_pyobject(py)?
                                    .into_any(),
                            ),
                        ],
                    )?,
                )?;
                return Ok(());
            }
            Err(err) => return Err(err),
        }
    }
}
#[pyfunction]
fn prep_cancelled(
    owner: &Bound<'_, PyAny>,
    current_id: &Bound<'_, PyAny>,
    following_id: &Bound<'_, PyAny>,
    expected: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if owner
        .getattr("stop")
        .and_then(|stop| stop.call_method0("is_set"))
        .and_then(|value| value.is_truthy())
        .unwrap_or(false)
    {
        return Ok(true);
    }
    let (running, pending) = queue(owner)?;
    let active = if running.len()? > 0 {
        running.get_item(0)?.get_item(1)?
    } else {
        owner.py().None().into_bound(owner.py())
    };
    let mut selected = None;
    for jobs in [&running, &pending] {
        for item in jobs.try_iter()? {
            let item = item?;
            if item.get_item(1)?.eq(following_id)? {
                selected = Some(item);
                break;
            }
        }
    }
    let Some(selected) = selected else {
        return Ok(true);
    };
    let plan = plan(owner)?;
    if !plan
        .call_method1(
            "graph_contract",
            (plan.call_method1("job_graph", (selected,))?,),
        )?
        .eq(plan.call_method1("graph_contract", (expected,))?)?
    {
        return Ok(true);
    }
    if active.eq(following_id)? {
        return Ok(false);
    }
    if !owner
        .call_method1("prompt_present", (current_id,))?
        .is_truthy()?
    {
        return Ok(true);
    }
    Ok(pending.len()? == 0
        || !plan
            .call_method1("head", (pending,))?
            .get_item(1)?
            .eq(following_id)?)
}
#[pyfunction]
fn prep_local_peer(owner: &Bound<'_, PyAny>, url: &str) -> PyResult<bool> {
    let parsed = owner
        .py()
        .import("urllib.parse")?
        .call_method1("urlsplit", (url,))?;
    let host = parsed
        .getattr("hostname")?
        .extract::<String>()
        .unwrap_or_default();
    Ok(["localhost", "127.0.0.1", "::1"].contains(&host.as_str())
        && parsed
            .getattr("port")?
            .eq(owner.getattr("server")?.getattr("port")?)?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prep_queue, m)?)?;
    m.add_function(wrap_pyfunction!(prep_record, m)?)?;
    m.add_function(wrap_pyfunction!(prep_status, m)?)?;
    m.add_function(wrap_pyfunction!(prep_eligible, m)?)?;
    m.add_function(wrap_pyfunction!(prep_job_present, m)?)?;
    m.add_function(wrap_pyfunction!(prep_prompt_present, m)?)?;
    m.add_function(wrap_pyfunction!(prep_decoder, m)?)?;
    m.add_function(wrap_pyfunction!(prep_tick, m)?)?;
    m.add_function(wrap_pyfunction!(prep_prepare_next, m)?)?;
    m.add_function(wrap_pyfunction!(prep_encoder_started, m)?)?;
    m.add_function(wrap_pyfunction!(prep_next_clip_missing, m)?)?;
    m.add_function(wrap_pyfunction!(prep_ensure_next_before_decode, m)?)?;
    m.add_function(wrap_pyfunction!(prep_cancelled, m)?)?;
    m.add_function(wrap_pyfunction!(prep_local_peer, m)?)?;
    Ok(())
}

//! One-shot prefetch and interruptible VAE preparation ownership.
use pyo3::{
    prelude::*,
    types::{PyCFunction, PyDict, PyTuple},
};
pub(super) fn with_context<T>(
    context: &Bound<'_, PyAny>,
    body: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    let py = context.py();
    context.call_method0("__enter__")?;
    match body() {
        Ok(value) => {
            context.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
            Ok(value)
        }
        Err(err) => {
            context.call_method1(
                "__exit__",
                (err.get_type(py), err.value(py), err.traceback(py)),
            )?;
            Err(err)
        }
    }
}
fn record(owner: &Bound<'_, PyAny>, state: &str, fields: &Bound<'_, PyDict>) -> PyResult<()> {
    owner.call_method("record", (state,), Some(fields))?;
    Ok(())
}
fn fields<'py>(
    py: Python<'py>,
    prompt: &Bound<'py, PyAny>,
    node: Option<&Bound<'py, PyAny>>,
    peer: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let fields = PyDict::new(py);
    fields.set_item("prompt_id", prompt)?;
    if let Some(node) = node {
        fields.set_item("node_id", node)?;
    }
    if let Some(peer) = peer {
        fields.set_item("peer", peer)?;
    }
    Ok(fields)
}
fn drop_attempt(owner: &Bound<'_, PyAny>, token: &Bound<'_, PyAny>) -> PyResult<()> {
    match owner.getattr("done")?.call_method1("remove", (token,)) {
        Ok(_) => Ok(()),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyValueError>(owner.py()) => Ok(()),
        Err(err) => Err(err),
    }
}
#[pyfunction]
#[pyo3(signature=(source,owner,current,following,graph,node_id,node,values,token,*,allow_after_sampling=false))]
fn prep_prefetch(
    source: &Bound<'_, PyModule>,
    owner: &Bound<'_, PyAny>,
    current: &Bound<'_, PyAny>,
    following: &Bound<'_, PyAny>,
    graph: &Bound<'_, PyAny>,
    node_id: &Bound<'_, PyAny>,
    node: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    token: &Bound<'_, PyAny>,
    allow_after_sampling: bool,
) -> PyResult<()> {
    let py = owner.py();
    let expected = py.import("copy")?.call_method1(
        "deepcopy",
        (source.getattr("graph_contract")?.call1((graph,))?,),
    )?;
    let contract = source
        .getattr("registered")?
        .call0()?
        .get_item(node.get_item("class_type")?)?
        .get_item(1)?;
    let values = contract.getattr("inputs")?.call1((values,))?;
    let url = values.get_item("server_url")?;
    let kw = PyDict::new(py);
    kw.set_item("speculative", true)?;
    let gate = source
        .getattr("gate")?
        .call1((&url,))?
        .call_method("enter", (), Some(&kw))?;
    with_context(&gate, || {
        if owner.getattr("done")?.contains(token)?
            || owner
                .call_method1(
                    "preparation_cancelled",
                    (current.get_item(1)?, following.get_item(1)?, &expected),
                )?
                .is_truthy()?
        {
            return Ok(());
        }
        if !allow_after_sampling
            && !owner
                .call_method1("eligible", (current.get_item(1)?,))?
                .is_truthy()?
        {
            return Ok(());
        }
        let scheduler = owner.getattr("scheduler")?;
        if scheduler.is_truthy()? {
            let kw = PyDict::new(py);
            kw.set_item("needs_clip", true)?;
            kw.set_item("decoder", owner.call_method1("decoder", (current,))?)?;
            scheduler.call_method(
                "decide",
                (current.get_item(1)?, following.get_item(1)?),
                Some(&kw),
            )?;
            scheduler.call_method1("invalidate_vae", (current.get_item(1)?, &url))?;
        }
        owner.getattr("done")?.call_method1("append", (token,))?;
        record(
            owner,
            "conditioning",
            &fields(py, &following.get_item(1)?, Some(node_id), Some(&url))?,
        )?;
        let started: f64 = source
            .getattr("time")?
            .call_method0("monotonic")?
            .extract()?;
        let attempt = (|| {
            let owner_callback = owner.clone().unbind();
            let current_callback = current.get_item(1)?.unbind();
            let following_callback = following.get_item(1)?.unbind();
            let expected_callback = expected.clone().unbind();
            let check = PyCFunction::new_closure(
                py,
                None,
                None,
                move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<bool> {
                    let py = args.py();
                    owner_callback
                        .bind(py)
                        .call_method1(
                            "preparation_cancelled",
                            (
                                current_callback.bind(py),
                                following_callback.bind(py),
                                expected_callback.bind(py),
                            ),
                        )?
                        .is_truthy()
                },
            )?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(
                "caller_interrupt",
                allow_after_sampling
                    && source
                        .getattr("caller_interrupt_enabled")?
                        .call0()?
                        .is_truthy()?,
            )?;
            let speculative = source.getattr("speculative_call")?.call0()?;
            let cancellation = source
                .getattr("cancellation_scope")?
                .call((check,), Some(&kwargs))?;
            with_context(&speculative, || {
                with_context(&cancellation, || {
                    let encoder = owner
                        .getattr("encoders")?
                        .get_item(node.get_item("class_type")?)?;
                    let result =
                        encoder
                            .call0()?
                            .call_method("run", (), Some(values.cast::<PyDict>()?))?;
                    source.getattr("check_cancelled")?.call0()?;
                    let handoffs = owner.getattr("handoffs")?;
                    let stored = if handoffs.is_truthy()? {
                        let normalized = source
                            .getattr("call_values")?
                            .call1((encoder.getattr("run")?, &values))?;
                        let handoff_module = py.import(
                            format!(
                                "{}.handoff",
                                source.name()?.to_str()?.rsplit_once('.').unwrap().0
                            )
                            .as_str(),
                        )?;
                        let kw = PyDict::new(py);
                        kw.set_item("max_bytes", handoffs.getattr("limit")?)?;
                        let blob = handoff_module
                            .getattr("pack")?
                            .call((&result,), Some(&kw))?;
                        source.getattr("check_cancelled")?.call0()?;
                        let stored = !blob.is_none()
                            && handoffs
                                .call_method1(
                                    "put",
                                    (
                                        (following.get_item(1)?, node_id),
                                        handoff_module
                                            .getattr("signature")?
                                            .call1((normalized,))?,
                                        blob,
                                    ),
                                )?
                                .is_truthy()?;
                        if !stored {
                            let fields = fields(py, &following.get_item(1)?, Some(node_id), None)?;
                            fields.set_item("reason", "handoff_too_large")?;
                            record(owner, "skipped", &fields)?;
                        }
                        stored
                    } else {
                        true
                    };
                    if scheduler.is_truthy()? {
                        let now: f64 = source
                            .getattr("time")?
                            .call_method0("monotonic")?
                            .extract()?;
                        scheduler.call_method1("observe_encoder", (now - started,))?;
                    }
                    Ok(stored)
                })
            })
        })();
        let stored = match attempt {
            Ok(stored) => stored,
            Err(err) => {
                drop_attempt(owner, token)?;
                if err
                    .value(py)
                    .is_instance(&source.getattr("RequestCancelled")?)?
                {
                    let fields = fields(py, &following.get_item(1)?, Some(node_id), None)?;
                    fields.set_item("reason", "queue_changed")?;
                    record(owner, "cancelled", &fields)?;
                }
                return Err(err);
            }
        };
        if stored {
            record(
                owner,
                "conditioning_ready",
                &fields(py, &following.get_item(1)?, Some(node_id), Some(&url))?,
            )?;
        }
        Ok(())
    })
}
#[pyfunction]
fn prep_prepare_vae(
    source: &Bound<'_, PyModule>,
    owner: &Bound<'_, PyAny>,
    current: &Bound<'_, PyAny>,
    decoder: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = owner.py();
    if !decoder.is_truthy()?
        || !owner
            .call_method1("eligible", (current.get_item(1)?,))?
            .is_truthy()?
    {
        return Ok(());
    }
    let node_id = decoder.get_item(0)?;
    let url = decoder.get_item(1)?;
    let model = decoder.get_item(2)?;
    let kind = decoder.get_item(3)?;
    let token = PyTuple::new(py, [current.get_item(1)?, node_id.clone(), model.clone()])?;
    if owner.call_method1("local_peer", (&url,))?.is_truthy()? {
        return Ok(());
    }
    let scheduler = owner.getattr("scheduler")?;
    if scheduler.is_truthy()? {
        if scheduler
            .call_method1("reserved", (current.get_item(1)?, decoder))?
            .is_truthy()?
        {
            return Ok(());
        }
    } else if owner.getattr("done")?.contains(&token)? {
        return Ok(());
    }
    if !owner.getattr("available")?.call1((&url,))?.is_truthy()? {
        let fields = fields(py, &current.get_item(1)?, None, None)?;
        fields.set_item("reason", "peer_busy")?;
        record(owner, "deferred", &fields)?;
        return Ok(());
    }
    let kw = PyDict::new(py);
    kw.set_item("speculative", true)?;
    let gate = source
        .getattr("gate")?
        .call1((&url,))?
        .call_method("enter", (), Some(&kw))?;
    with_context(&gate, || {
        if !owner
            .call_method1("eligible", (current.get_item(1)?,))?
            .is_truthy()?
            || owner
                .call_method1("next_clip_missing", (current.get_item(1)?, &url))?
                .is_truthy()?
        {
            return Ok(());
        }
        let event_fields = fields(py, &current.get_item(1)?, Some(&node_id), Some(&url))?;
        event_fields.set_item("model", &model)?;
        for (name, key) in [
            ("total_steps", "total"),
            ("remaining_steps", "remaining_steps"),
        ] {
            event_fields.set_item(
                name,
                if scheduler.is_truthy()? {
                    scheduler.getattr("last")?.call_method1("get", (key,))?
                } else {
                    py.None().into_bound(py)
                },
            )?;
        }
        record(owner, "vae", &event_fields)?;
        let started: f64 = source
            .getattr("time")?
            .call_method0("monotonic")?
            .extract()?;
        let owner_callback = owner.clone().unbind();
        let current_callback = current.get_item(1)?.unbind();
        let url_callback = url.clone().unbind();
        let check = PyCFunction::new_closure(
            py,
            None,
            None,
            move |args: &Bound<'_, PyTuple>, _: Option<&Bound<'_, PyDict>>| -> PyResult<bool> {
                let py = args.py();
                let owner = owner_callback.bind(py);
                Ok(!owner
                    .call_method1("prompt_present", (current_callback.bind(py),))?
                    .is_truthy()?
                    || owner
                        .call_method1(
                            "next_clip_missing",
                            (current_callback.bind(py), url_callback.bind(py)),
                        )?
                        .is_truthy()?)
            },
        )?;
        let kw = PyDict::new(py);
        kw.set_item("caller_interrupt", false)?;
        let cancellation = source
            .getattr("cancellation_scope")?
            .call((check,), Some(&kw))?;
        if let Err(err) = with_context(&cancellation, || {
            owner.getattr("warmup")?.call1((&url, &model, &kind))?;
            source.getattr("check_cancelled")?.call0()?;
            Ok(())
        }) {
            if err
                .value(py)
                .is_instance(&source.getattr("RequestCancelled")?)?
            {
                let switched = owner
                    .call_method1("prompt_present", (current.get_item(1)?,))?
                    .is_truthy()?
                    && owner
                        .call_method1("next_clip_missing", (current.get_item(1)?, &url))?
                        .is_truthy()?;
                let fields = fields(py, &current.get_item(1)?, Some(&node_id), None)?;
                fields.set_item(
                    "reason",
                    if switched {
                        "next_clip_priority"
                    } else {
                        "queue_changed"
                    },
                )?;
                record(
                    owner,
                    if switched {
                        "vae_deferred"
                    } else {
                        "cancelled"
                    },
                    &fields,
                )?;
            }
            return Err(err);
        }
        owner.getattr("done")?.call_method1("append", (&token,))?;
        if scheduler.is_truthy()? {
            let now: f64 = source
                .getattr("time")?
                .call_method0("monotonic")?
                .extract()?;
            scheduler.call_method1("observe_vae", (now - started,))?;
            scheduler.call_method1("reserve", (current.get_item(1)?, decoder))?;
        }
        let fields = fields(py, &current.get_item(1)?, Some(&node_id), Some(&url))?;
        fields.set_item("model", &model)?;
        record(owner, "vae_ready", &fields)?;
        Ok(())
    })
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prep_prefetch, m)?)?;
    m.add_function(wrap_pyfunction!(prep_prepare_vae, m)?)?;
    Ok(())
}

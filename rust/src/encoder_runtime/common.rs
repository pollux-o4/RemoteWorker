use pyo3::{prelude::*, types::*};
pub(super) fn field<'py>(v: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    v.get_item(name)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_owned()))
}
pub(super) fn selected<'py>(
    v: &Bound<'py, PyDict>,
    names: &[&str],
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(v.py());
    for n in names {
        out.set_item(*n, field(v, n)?)?;
    }
    Ok(out)
}
pub(super) fn call<'py>(
    m: &Bound<'py, PyAny>,
    name: &str,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    m.getattr(name)?.call((), Some(kwargs))
}
pub(super) fn uuid(m: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(m.getattr("uuid")?
        .call_method0("uuid4")?
        .getattr("hex")?
        .unbind())
}
pub(super) fn clip<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
    qwen: bool,
    conditioning: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let first = field(
        v,
        if qwen || conditioning {
            "remote_text_encoder"
        } else {
            "clip_name"
        },
    )?;
    if qwen {
        m.getattr("ClipSelection")?
            .call1((first, "qwen_image", "default"))
    } else {
        m.getattr("ClipSelection")?
            .call1((first, field(v, "clip_type")?, field(v, "device")?))
    }
}
pub(super) fn api<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    m.getattr("HttpComfyApi")?
        .call1((field(v, "server_url")?, field(v, "timeout_seconds")?))
}
pub(super) fn validate_policy(
    m: &Bound<'_, PyAny>,
    v: &Bound<'_, PyDict>,
    name: &str,
    allowed: &str,
) -> PyResult<()> {
    m.getattr("_validate_policy")?
        .call1((field(v, name)?, m.getattr(allowed)?, name))?;
    Ok(())
}
pub(super) fn decode_conditioning<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("map_location", "cpu")?;
    kwargs.set_item("weights_only", false)?;
    let stream = py.import("io")?.getattr("BytesIO")?.call1((data,))?;
    py.import("torch")?
        .getattr("load")?
        .call((stream,), Some(&kwargs))
}

pub(super) fn with_context<'py, F>(
    context: &Bound<'py, PyAny>,
    operation: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
{
    let entered = context.call_method0("__enter__")?;
    match operation(&entered) {
        Ok(value) => {
            context.call_method1(
                "__exit__",
                (
                    context.py().None(),
                    context.py().None(),
                    context.py().None(),
                ),
            )?;
            Ok(value)
        }
        Err(error) => {
            let py = context.py();
            let suppressed = context
                .call_method1(
                    "__exit__",
                    (error.get_type(py), error.value(py), error.traceback(py)),
                )?
                .is_truthy()?;
            if suppressed {
                Ok(py.None().into_bound(py))
            } else {
                Err(error)
            }
        }
    }
}
pub(super) fn release_preflight(
    m: &Bound<'_, PyAny>,
    api: &Bound<'_, PyAny>,
    policy: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if policy.extract::<String>()? == "release_after_result" {
        m.getattr("validate_cleanup_capability")?
            .call1((api.call_method1("require_node", ("VRAMCleanup",))?,))?;
    }
    Ok(())
}
pub(super) fn upload_file<'py>(
    request: &Bound<'py, PyAny>,
    filename: &str,
    data: &Bound<'py, PyAny>,
    content_type: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let args = PyDict::new(request.py());
    args.set_item("filename", filename)?;
    args.set_item("data", data)?;
    args.set_item("content_type", content_type)?;
    request.call_method("upload_file", (), Some(&args))
}
pub(super) fn upload_image<'py>(
    m: &Bound<'py, PyAny>,
    request: &Bound<'py, PyAny>,
    index: usize,
    image: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let args = PyDict::new(m.py());
    args.set_item("index", index)?;
    args.set_item("png", m.getattr("encode_image_frame_png")?.call1((image,))?)?;
    request.call_method("upload_image", (), Some(&args))
}
pub(super) fn artifacts<'py, F>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
    workflow: &Bound<'py, PyAny>,
    specs: Vec<Bound<'py, PyAny>>,
    consume: F,
) -> PyResult<Bound<'py, PyAny>>
where
    F: FnOnce(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
{
    let specs = PyTuple::new(m.py(), specs)?;
    let context = m.getattr("completed_artifacts")?.call1((
        field(v, "api")?,
        workflow,
        field(v, "request_id")?,
        specs,
    ))?;
    with_context(&context, consume)
}

use super::*;
use pyo3::{prelude::*, types::*};
pub(super) fn qwen_run<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    validate_policy(m, v, "peer_memory_policy", "PEER_MEMORY_POLICIES")?;
    validate_policy(m, v, "cache_policy", "CACHE_POLICIES")?;
    let conn = m.getattr("_validate_connection")?.call1((
        field(v, "server_url")?,
        field(v, "timeout_seconds")?,
        field(v, "remote_text_encoder")?,
        field(v, "vae_name")?,
    ))?;
    let normalized = v.copy()?;
    for (i, key) in [
        "server_url",
        "timeout_seconds",
        "remote_text_encoder",
        "vae_name",
    ]
    .iter()
    .enumerate()
    {
        normalized.set_item(*key, conn.get_item(i)?)?;
    }
    let images = field(v, "images")?.cast_into::<PyDict>()?;
    let frames = call(m, "_ordered_qwen_image_frames", &images)?;
    let clip = clip(m, &normalized, true, false)?;
    m.getattr("_validate_choice")?
        .call1(("prompt", field(v, "prompt")?))?;
    if !field(v, "negative_prompt")?.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyValueError::new_err(
            "Remote Qwen Image 2.1 negative_prompt must be a string",
        ));
    }
    let cache = field(v, "cache_policy")?.extract::<String>()? == "reuse_same_inputs";
    let args = selected(
        &normalized,
        &[
            "vae_name",
            "prompt",
            "negative_prompt",
            "resolution",
            "negative_vision",
            "server_url",
        ],
    )?;
    args.set_item("clip", &clip)?;
    args.set_item("frames", &frames)?;
    let key = call(m, "_cache_key", &args)?;
    let cache_store = m.getattr("_CACHE")?;
    if cache {
        let cached = cache_store.call_method1("get", (&key,))?;
        if !cached.is_none() {
            return m.getattr("_decode")?.call(cached.cast::<PyTuple>()?, None);
        }
    }
    args.del_item("server_url")?;
    args.set_item("api", api(m, &normalized)?)?;
    args.set_item("request_id", uuid(m)?)?;
    args.set_item("peer_memory_policy", field(v, "peer_memory_policy")?)?;
    let result = call(m, "execute", &args)?;
    if cache {
        cache_store.call_method1("put", (key, &result))?;
    }
    m.getattr("_decode")?.call(result.cast::<PyTuple>()?, None)
}

pub(super) fn qwen_execute<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let api = field(v, "api")?;
    let info = api.call_method1("require_node", (m.getattr("QWEN_WORKER")?,))?;
    m.getattr("_validate_qwen_capability")?.call1((info,))?;
    let normalized = m.getattr("_validate_peer_loaders")?.call1((
        &api,
        field(v, "clip")?,
        field(v, "vae_name")?,
    ))?;
    release_preflight(m, &api, &field(v, "peer_memory_policy")?)?;
    let request = m
        .getattr("RequestInputs")?
        .call1((&api, field(v, "request_id")?))?;
    with_context(&request, |request| {
        let refs = PyList::empty(m.py());
        for (i, frame) in field(v, "frames")?.try_iter()?.enumerate() {
            refs.append(upload_image(m, request, i + 1, &frame?)?)?;
        }
        let args = selected(
            v,
            &[
                "prompt",
                "negative_prompt",
                "resolution",
                "negative_vision",
                "request_id",
                "peer_memory_policy",
            ],
        )?;
        args.set_item("clip", normalized.get_item(0)?)?;
        args.set_item("vae_name", normalized.get_item(1)?)?;
        args.set_item("image_refs", refs)?;
        let workflow = call(m, "build_workflow", &args)?;
        artifacts(
            m,
            v,
            &workflow,
            vec![
                m.getattr("POSITIVE_ARTIFACT")?,
                m.getattr("NEGATIVE_ARTIFACT")?,
                m.getattr("LATENT_ARTIFACT")?,
            ],
            |blobs| {
                Ok(PyTuple::new(
                    m.py(),
                    [
                        blobs.get_item("positive")?,
                        blobs.get_item("negative")?,
                        blobs.get_item("latent")?,
                    ],
                )?
                .into_any())
            },
        )
    })
}
#[pyfunction]
#[pyo3(signature=(clip,prompt,negative_prompt,resolution,vae,negative_vision,images))]
fn qwen_encode_native(
    clip: &Bound<'_, PyAny>,
    prompt: &Bound<'_, PyAny>,
    negative_prompt: &Bound<'_, PyAny>,
    resolution: &Bound<'_, PyAny>,
    vae: Option<&Bound<'_, PyAny>>,
    negative_vision: bool,
    images: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let py = clip.py();
    let resolution =
        crate::encoder_policy::qwen_worker_inputs_native(prompt, negative_prompt, resolution)?;
    let management = py.import("comfy.model_management")?;
    let utils = py.import("comfy.utils")?;
    let helpers = py.import("node_helpers")?;
    let torch = py.import("torch")?;
    let refs = PyList::empty(py);
    let visuals = PyList::empty(py);
    let mut latent_width = if resolution == 0 { 1024 } else { resolution };
    let mut latent_height = latent_width;
    let mut ordered = Vec::new();
    for (key, image) in images.iter() {
        let name = key.extract::<String>()?;
        let suffix = name.rsplit('_').next().unwrap_or("");
        let index = suffix
            .parse::<i64>()
            .map_err(|_| PyValueError::new_err("Invalid Qwen image socket"))?;
        ordered.push((index, image));
    }
    ordered.sort_by_key(|(index, _)| *index);
    for (_, image) in ordered {
        if image.is_none() {
            continue;
        }
        let first = image.get_item(pyo3::types::PySlice::new(py, 0, 1, 1))?;
        let samples = first.call_method1("movedim", (-1, 1))?;
        let shape = samples.getattr("shape")?;
        let source_width = shape.get_item(3)?.extract::<i64>()?;
        let source_height = shape.get_item(2)?.extract::<i64>()?;
        let (width, height) =
            crate::encoder_policy::qwen_resize_native(source_width, source_height, resolution)?;
        let resized = if width == source_width && height == source_height {
            first
        } else {
            utils
                .getattr("common_upscale")?
                .call1((samples, width, height, "lanczos", "disabled"))?
                .call_method1("movedim", (1, -1))?
        };
        if visuals.is_empty() {
            latent_width = width;
            latent_height = height;
        }
        let select = |start: isize, stop: isize| -> PyResult<Bound<'_, PyAny>> {
            let slices = PyTuple::new(
                py,
                [
                    pyo3::types::PySlice::new(py, 0, isize::MAX, 1),
                    pyo3::types::PySlice::new(py, 0, isize::MAX, 1),
                    pyo3::types::PySlice::new(py, 0, isize::MAX, 1),
                    pyo3::types::PySlice::new(py, start, stop, 1),
                ],
            )?;
            resized.get_item(slices)
        };
        let mut rgb = select(0, 3)?;
        let channel_count = resized.getattr("shape")?.get_item(-1)?.extract::<i64>()?;
        if channel_count > 3 {
            let alpha = select(3, isize::MAX)?;
            let scaled = rgb.call_method1("__mul__", (&alpha,))?;
            let background = alpha.call_method1("__rsub__", (1.0,))?;
            rgb = scaled.call_method1("__add__", (background,))?;
        }
        visuals.append(rgb)?;
        if let Some(vae) = vae.filter(|v| !v.is_none()) {
            refs.append(vae.call_method1("encode", (&resized,))?)?;
        }
    }
    let keep_vision = refs.is_empty();
    let positive_kwargs = PyDict::new(py);
    positive_kwargs.set_item("images", &visuals)?;
    positive_kwargs.set_item("keep_vision", keep_vision)?;
    positive_kwargs.set_item("prevent_empty_text", true)?;
    let positive_tokens = clip.call_method("tokenize", (prompt,), Some(&positive_kwargs))?;
    let negative_kwargs = positive_kwargs.copy()?;
    if !negative_vision {
        negative_kwargs.set_item("images", PyList::empty(py))?;
    }
    let negative_tokens =
        clip.call_method("tokenize", (negative_prompt,), Some(&negative_kwargs))?;
    let mut positive = clip.call_method1("encode_from_tokens_scheduled", (positive_tokens,))?;
    let mut negative = clip.call_method1("encode_from_tokens_scheduled", (negative_tokens,))?;
    if !refs.is_empty() {
        let values = PyDict::new(py);
        values.set_item("reference_latents", refs)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("append", true)?;
        positive = helpers
            .getattr("conditioning_set_values")?
            .call((positive, &values), Some(&kwargs))?;
        negative = helpers
            .getattr("conditioning_set_values")?
            .call((negative, &values), Some(&kwargs))?;
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("device", management.call_method0("intermediate_device")?)?;
    let latent = torch.getattr("zeros")?.call(
        ([1, 64, latent_height / 16, latent_width / 16],),
        Some(&kwargs),
    )?;
    let result = PyDict::new(py);
    result.set_item("samples", latent)?;
    Ok(PyTuple::new(py, [positive, negative, result.into_any()])?
        .into_any()
        .unbind())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qwen_encode_native, m)?)?;
    Ok(())
}

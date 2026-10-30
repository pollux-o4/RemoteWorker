//! Encoder execution policy; Python callbacks provide Comfy/PyTorch and IO adapters.
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
fn field<'py>(v: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    v.get_item(name)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_owned()))
}
fn selected<'py>(v: &Bound<'py, PyDict>, names: &[&str]) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(v.py());
    for n in names {
        out.set_item(*n, field(v, n)?)?;
    }
    Ok(out)
}
fn call<'py>(
    m: &Bound<'py, PyAny>,
    name: &str,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    m.getattr(name)?.call((), Some(kwargs))
}
fn uuid(m: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(m.getattr("uuid")?
        .call_method0("uuid4")?
        .getattr("hex")?
        .unbind())
}
fn clip<'py>(
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
fn api<'py>(m: &Bound<'py, PyAny>, v: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
    m.getattr("HttpComfyApi")?
        .call1((field(v, "server_url")?, field(v, "timeout_seconds")?))
}
fn validate_policy(
    m: &Bound<'_, PyAny>,
    v: &Bound<'_, PyDict>,
    name: &str,
    allowed: &str,
) -> PyResult<()> {
    m.getattr("_validate_policy")?
        .call1((field(v, name)?, m.getattr(allowed)?, name))?;
    Ok(())
}
fn decode_conditioning<'py>(
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
fn qwen_run<'py>(m: &Bound<'py, PyAny>, v: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
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
fn conditioning_run<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let strategy = field(v, "conditioning_strategy")?.extract::<String>()?;
    let frames = if strategy == "text" {
        PyList::empty(m.py()).into_any()
    } else {
        let images = field(v, "images")?.cast_into::<PyDict>()?;
        m.getattr("ordered_image_frames")?
            .call((field(v, "image_count")?,), Some(&images))?
    };
    let clip = clip(m, v, false, true)?;
    let cache = field(v, "cache_policy")?.extract::<String>()? == "reuse_same_inputs";
    let args = selected(
        v,
        &[
            "prompt",
            "grounding_px",
            "system_prompt",
            "server_url",
            "peer_memory_policy",
        ],
    )?;
    args.set_item("strategy_name", &strategy)?;
    args.set_item("clip", clip)?;
    args.set_item("frames", frames)?;
    let key = if cache {
        let key = call(m, "_conditioning_cache_key", &args)?;
        let cached = m
            .getattr("_get_cached_conditioning_bytes")?
            .call1((&key,))?;
        if !cached.is_none() {
            return Ok(PyTuple::new(m.py(), [decode_conditioning(m.py(), &cached)?])?.into_any());
        }
        Some(key)
    } else {
        validate_policy(m, v, "cache_policy", "CACHE_POLICIES")?;
        None
    };
    args.del_item("server_url")?;
    args.set_item("api", api(m, v)?)?;
    args.set_item("request_id", uuid(m)?)?;
    let data = call(m, "execute_strategy_bytes", &args)?;
    if let Some(key) = key {
        m.getattr("_put_cached_conditioning_bytes")?
            .call1((key, &data))?;
    }
    Ok(PyTuple::new(m.py(), [decode_conditioning(m.py(), &data)?])?.into_any())
}
#[pyclass]
struct DecodeCapture {
    module: Py<PyAny>,
    captured: Option<(Py<PyAny>, Py<PyAny>)>,
}
#[pymethods]
impl DecodeCapture {
    fn __call__(
        &mut self,
        py: Python<'_>,
        conditioning: &Bound<'_, PyAny>,
        latent: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bytes = py.import("builtins")?.getattr("bytes")?;
        let a = bytes.call1((conditioning,))?.unbind();
        let b = bytes.call1((latent,))?.unbind();
        self.captured = Some((a, b));
        Ok(self
            .module
            .bind(py)
            .getattr("decode_remote_outputs")?
            .call1((conditioning, latent))?
            .unbind())
    }
}
fn minimax_run<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
    reference: bool,
) -> PyResult<Bound<'py, PyAny>> {
    m.getattr("validate_cache_policy")?
        .call1((field(v, "cache_policy")?,))?;
    let media = if reference {
        field(v, "media")?.cast_into::<PyDict>()?.copy()?
    } else {
        PyDict::new(m.py())
    };
    let normalized = v.copy()?;
    for (name, override_name) in if reference {
        vec![
            ("vae_name", "vae_name_override"),
            ("audio_vae_name", "audio_vae_name_override"),
        ]
    } else {
        vec![("vae_name", "vae_name_override")]
    } {
        let override_value = if reference {
            let value = media
                .get_item(override_name)?
                .unwrap_or_else(|| m.py().None().into_bound(m.py()));
            media.del_item(override_name).ok();
            value
        } else {
            field(v, override_name)?
        };
        let kwargs = selected(v, &["unique_id", "extra_pnginfo"])?;
        kwargs.set_item(
            "input_name",
            if override_value.is_none() {
                name
            } else {
                override_name
            },
        )?;
        let value = m
            .getattr("resolve_vae_name")?
            .call((field(v, name)?, override_value), Some(&kwargs))?;
        normalized.set_item(name, value)?;
    }
    let clip = clip(m, v, false, false)?;
    let args = selected(
        &normalized,
        &[
            "vae_name",
            "prompt",
            "width",
            "height",
            "length",
            "peer_memory_policy",
        ],
    )?;
    args.set_item("clip", clip)?;
    if reference {
        args.set_item("audio_vae_name", field(&normalized, "audio_vae_name")?)?;
        args.set_item("ref_image_size", field(v, "ref_image_size")?)?;
        for (key, prefix, batches, max) in [
            ("images", "ref_image", false, 9),
            ("videos", "ref_video", true, 3),
        ] {
            let kwargs = PyDict::new(m.py());
            kwargs.set_item("batches", batches)?;
            kwargs.set_item("maximum", max)?;
            let items = m
                .getattr("_ordered_images")?
                .call((prefix, &media), Some(&kwargs))?;
            args.set_item(key, items)?;
        }
        for (key, prefix) in [("video_audios", "ref_video_audio"), ("audios", "ref_audio")] {
            args.set_item(
                key,
                m.getattr("_ordered_audio")?.call1((prefix, &media, 3))?,
            )?;
        }
        if field(&args, "video_audios")?.len()? > field(&args, "videos")?.len()? {
            return Err(PyValueError::new_err(
                "Remote MiniMax H3 video audio requires a same-index reference video",
            ));
        }
    } else {
        args.set_item("first_frame", field(v, "first_frame")?)?;
        args.set_item("last_frame", field(v, "last_frame")?)?;
    }
    let cache = field(v, "cache_policy")?.extract::<String>()? == "reuse_same_inputs";
    let key = if cache {
        args.set_item("server_url", field(v, "server_url")?)?;
        let key = call(
            m,
            if reference {
                "_reference_cache_key"
            } else {
                "_i2v_cache_key"
            },
            &args,
        )?;
        args.del_item("server_url")?;
        let cached = m.getattr("get_cached_minimax_outputs")?.call1((&key,))?;
        if !cached.is_none() {
            return m
                .getattr("decode_remote_outputs")?
                .call(cached.cast::<PyTuple>()?, None);
        }
        Some(key)
    } else {
        None
    };
    let capture = Py::new(
        m.py(),
        DecodeCapture {
            module: m.clone().unbind(),
            captured: None,
        },
    )?;
    args.set_item("deserialize", capture.bind(m.py()))?;
    args.set_item("request_id", uuid(m)?)?;
    args.set_item("api", api(m, v)?)?;
    let result = call(m, "execute", &args)?;
    if let Some(key) = key {
        if let Some((a, b)) = &capture.borrow(m.py()).captured {
            m.getattr("put_cached_minimax_outputs")?
                .call1((key, a, b))?;
        }
    }
    Ok(result)
}
#[pyfunction]
fn encoder_run_native(
    kind: &str,
    module: &Bound<'_, PyAny>,
    values: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let result = match kind {
        "qwen" => qwen_run(module, values),
        "prompt" => prompt_run(module, values),
        "conditioning" => conditioning_run(module, values),
        "minimax-reference" => minimax_run(module, values, true),
        "minimax-i2v" => minimax_run(module, values, false),
        _ => return Err(PyValueError::new_err("Unknown remote encoder kind")),
    };
    match result {
        Ok(result) => Ok(result.unbind()),
        Err(error) => {
            let py = module.py();
            if kind != "qwen"
                && (error.is_instance_of::<PyOSError>(py)
                    || error.is_instance_of::<PyRuntimeError>(py)
                    || error.is_instance_of::<PyTimeoutError>(py)
                    || error.is_instance_of::<PyValueError>(py))
            {
                let prefix = match kind {
                    "conditioning" => "Remote Conditioning",
                    "prompt" => "Remote Prompt Enhancer",
                    "minimax-reference" => "Remote MiniMax H3",
                    _ => "Remote MiniMax H3 Image to Video",
                };
                let replacement =
                    PyRuntimeError::new_err(format!("{prefix} failed: {}", error.value(py).str()?));
                replacement.set_cause(py, Some(error));
                Err(replacement)
            } else {
                Err(error)
            }
        }
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encoder_run_native, m)?)?;
    m.add_function(wrap_pyfunction!(encoder_execute_native, m)?)?;
    m.add_function(wrap_pyfunction!(prompt_generate_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_encode_native, m)?)?;
    Ok(())
}

fn with_context<'py, F>(context: &Bound<'py, PyAny>, operation: F) -> PyResult<Bound<'py, PyAny>>
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
fn release_preflight(
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
fn upload_file<'py>(
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
fn upload_image<'py>(
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
fn artifacts<'py, F>(
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
fn conditioning_execute<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
    strategy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    validate_policy(m, v, "peer_memory_policy", "PEER_MEMORY_POLICIES")?;
    let api = field(v, "api")?;
    let name = if strategy {
        field(v, "strategy_name")?.extract::<String>()?
    } else {
        "text".into()
    };
    if name == "text" {
        if strategy && field(v, "frames")?.len()? > 0 {
            return Err(PyValueError::new_err(
                "Remote Conditioning text strategy requires 0 images",
            ));
        }
        release_preflight(m, &api, &field(v, "peer_memory_policy")?)?;
        let workflow = m.getattr("build_workflow")?.call1((
            field(v, "clip")?,
            field(v, "prompt")?,
            field(v, "request_id")?,
            field(v, "peer_memory_policy")?,
        ))?;
        return artifacts(
            m,
            v,
            &workflow,
            vec![m.getattr("CONDITIONING_ARTIFACT")?],
            |blobs| blobs.get_item("conditioning"),
        );
    }
    if name != "krea2-grounded" {
        return Err(PyValueError::new_err(format!(
            "Remote Conditioning strategy is not registered: {name:?}"
        )));
    }
    let frames = field(v, "frames")?;
    let count = frames.len()?;
    if !(1..=2).contains(&count) {
        return Err(PyValueError::new_err(format!(
            "Remote Conditioning {name} requires 1 to 2 images; got {count}"
        )));
    }
    let spec = m.getattr("STRATEGIES")?.get_item(&name)?;
    let capability = api.call_method1("require_node", (spec.getattr("remote_node")?,))?;
    m.getattr("validate_strategy_capability")?
        .call1((&spec, capability, count))?;
    release_preflight(m, &api, &field(v, "peer_memory_policy")?)?;
    let encoded = PyList::empty(m.py());
    for frame in frames.try_iter()? {
        encoded.append(m.getattr("encode_image_frame_png")?.call1((frame?,))?)?;
    }
    let request = m
        .getattr("RequestInputs")?
        .call1((&api, field(v, "request_id")?))?;
    with_context(&request, |request| {
        let refs = PyList::empty(m.py());
        for (i, png) in encoded.iter().enumerate() {
            let args = PyDict::new(m.py());
            args.set_item("index", i + 1)?;
            args.set_item("png", png)?;
            refs.append(request.call_method("upload_image", (), Some(&args))?)?;
        }
        let args = selected(
            v,
            &[
                "strategy_name",
                "clip",
                "prompt",
                "request_id",
                "grounding_px",
                "system_prompt",
                "peer_memory_policy",
            ],
        )?;
        args.set_item("image_refs", refs)?;
        let workflow = call(m, "build_strategy_workflow", &args)?;
        artifacts(
            m,
            v,
            &workflow,
            vec![m.getattr("CONDITIONING_ARTIFACT")?],
            |blobs| blobs.get_item("conditioning"),
        )
    })
}
fn qwen_execute<'py>(m: &Bound<'py, PyAny>, v: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
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
fn minimax_execute<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
    reference: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let api = field(v, "api")?;
    let info = api.call_method1(
        "require_node",
        (if reference {
            "MiniMaxH3ReferenceToVideo"
        } else {
            "MiniMaxH3ImageToVideo"
        },),
    )?;
    if reference {
        m.getattr("validate_ref2va_capability")?.call1((info,))?;
    } else {
        let outputs = info.call_method1("get", ("output", PyList::empty(m.py())))?;
        if !outputs.contains("CONDITIONING")? || !outputs.contains("LATENT")? {
            return Err(PyRuntimeError::new_err(
                "Remote MiniMax H3 Image to Video peer is incompatible",
            ));
        }
    }
    m.getattr("validate_release_cleanup_capability")?
        .call1((&api, field(v, "peer_memory_policy")?))?;
    if reference {
        m.getattr("validate_peer_inputs")?.call1((
            &api,
            field(v, "clip")?,
            field(v, "vae_name")?,
            field(v, "audio_vae_name")?,
        ))?;
    } else {
        m.getattr("validate_peer_inputs")?.call1((
            &api,
            field(v, "clip")?,
            field(v, "vae_name")?,
        ))?;
    }
    let request = m
        .getattr("RequestInputs")?
        .call1((&api, field(v, "request_id")?))?;
    with_context(&request, |request| {
        let args = selected(
            v,
            &[
                "clip",
                "vae_name",
                "prompt",
                "width",
                "height",
                "length",
                "request_id",
                "peer_memory_policy",
            ],
        )?;
        if reference {
            args.set_item("audio_vae_name", field(v, "audio_vae_name")?)?;
            args.set_item("ref_image_size", field(v, "ref_image_size")?)?;
            let images = PyList::empty(m.py());
            for (i, image) in field(v, "images")?.try_iter()?.enumerate() {
                images.append(upload_image(m, request, i + 1, &image?.get_item(0)?)?)?;
            }
            args.set_item("image_refs", images)?;
            let videos = PyList::empty(m.py());
            for (vi, video) in field(v, "videos")?.try_iter()?.enumerate() {
                let video = video?;
                let paths = PyList::empty(m.py());
                let count = video.getattr("shape")?.get_item(0)?.extract::<usize>()?;
                for fi in 0..count {
                    let data = m
                        .getattr("encode_image_frame_png")?
                        .call1((video.get_item(fi)?,))?;
                    paths.append(upload_file(
                        request,
                        &format!("video-{vi:02}-frame-{fi:04}.png"),
                        &data,
                        "image/png",
                    )?)?;
                }
                videos.append(paths)?;
            }
            args.set_item("video_refs", videos)?;
            for (source, target, prefix) in [
                ("video_audios", "video_audio_refs", "video-audio"),
                ("audios", "audio_refs", "audio"),
            ] {
                let refs = PyList::empty(m.py());
                for (i, audio) in field(v, source)?.try_iter()?.enumerate() {
                    let data = m.getattr("encode_audio_wav")?.call1((audio?,))?;
                    refs.append(upload_file(
                        request,
                        &format!("{prefix}-{i:02}.wav"),
                        &data,
                        "audio/wav",
                    )?)?;
                }
                args.set_item(target, refs)?;
            }
        } else {
            for name in ["first_frame", "last_frame"] {
                let image = field(v, name)?;
                let path = if image.is_none() {
                    m.py().None().into_bound(m.py())
                } else {
                    let frame = m.getattr("_single_frame")?.call1((image, name))?;
                    let data = m.getattr("encode_image_frame_png")?.call1((frame,))?;
                    upload_file(request, &format!("{name}.png"), &data, "image/png")?
                };
                args.set_item(name, path)?;
            }
        }
        let workflow = call(m, "build_workflow", &args)?;
        artifacts(
            m,
            v,
            &workflow,
            vec![
                m.getattr("CONDITIONING_ARTIFACT")?,
                m.getattr("LATENT_ARTIFACT")?,
            ],
            |blobs| {
                field(v, "deserialize")?
                    .call1((blobs.get_item("conditioning")?, blobs.get_item("latent")?))
            },
        )
    })
}
fn prompt_execute<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let api = field(v, "api")?;
    let profile = api.call_method1("get_prompt_enhancer_profile", (field(v, "profile_id")?,))?;
    if !profile.is_instance(&m.getattr("PromptEnhancerProfile")?)? {
        return Err(PyRuntimeError::new_err(
            "Remote Prompt Enhancer profile response is invalid",
        ));
    }
    api.call_method1("require_node", ("RemotePromptGenerate",))?;
    api.call_method1("require_node", ("SaveRemoteText",))?;
    let images = field(v, "images")?;
    let videos = field(v, "videos")?;
    let images = if images.is_none() {
        PyList::empty(m.py()).into_any()
    } else {
        images
    };
    let videos = if videos.is_none() {
        PyList::empty(m.py()).into_any()
    } else {
        videos
    };
    crate::encoder_policy::validate_prompt_media_native(&profile, images.len()?, &videos)?;
    let request = m
        .getattr("RequestInputs")?
        .call1((&api, field(v, "request_id")?))?;
    with_context(&request, |request| {
        let refs = PyList::empty(m.py());
        for (i, image) in images.try_iter()?.enumerate() {
            refs.append(upload_image(m, request, i + 1, &image?)?)?;
        }
        let video_refs = PyList::empty(m.py());
        for (vi, video) in videos.try_iter()?.enumerate() {
            let video = video?;
            let paths = PyList::empty(m.py());
            let count = video.getattr("shape")?.get_item(0)?.extract::<usize>()?;
            for fi in 0..count {
                let data = m
                    .getattr("encode_image_frame_png")?
                    .call1((video.get_item(fi)?,))?;
                paths.append(upload_file(
                    request,
                    &format!("prompt-video-{vi:02}-frame-{fi:03}.png"),
                    &data,
                    "image/png",
                )?)?;
            }
            video_refs.append(paths)?;
        }
        let kwargs = PyDict::new(m.py());
        kwargs.set_item("image_refs", refs)?;
        kwargs.set_item("video_refs", video_refs)?;
        let workflow = m.getattr("build_workflow")?.call(
            (&profile, field(v, "prompt")?, field(v, "request_id")?),
            Some(&kwargs),
        )?;
        artifacts(
            m,
            v,
            &workflow,
            vec![m.getattr("PROMPT_ARTIFACT")?],
            |blobs| {
                m.getattr("decode_generated_prompt")?
                    .call1((blobs.get_item("prompt")?,))
            },
        )
    })
}
#[pyfunction]
fn encoder_execute_native(
    kind: &str,
    module: &Bound<'_, PyAny>,
    values: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    Ok(match kind {
        "conditioning" => conditioning_execute(module, values, false)?,
        "conditioning-strategy" => conditioning_execute(module, values, true)?,
        "qwen" => qwen_execute(module, values)?,
        "minimax-reference" => minimax_execute(module, values, true)?,
        "minimax-i2v" => minimax_execute(module, values, false)?,
        "prompt" => prompt_execute(module, values)?,
        _ => return Err(PyValueError::new_err("Unknown remote execution kind")),
    }
    .unbind())
}

fn prompt_run<'py>(m: &Bound<'py, PyAny>, v: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
    let args = selected(v, &["profile_id", "prompt"])?;
    args.set_item("request_id", uuid(m)?)?;
    args.set_item(
        "api",
        m.getattr("PromptComfyApi")?
            .call1((field(v, "server_url")?, field(v, "timeout_seconds")?))?,
    )?;
    args.set_item(
        "images",
        m.getattr("_ordered_media")?
            .call1(("image", field(v, "media")?, 8))?,
    )?;
    args.set_item(
        "videos",
        m.getattr("_ordered_batches")?
            .call1(("video", field(v, "media")?, 3))?,
    )?;
    Ok(PyTuple::new(m.py(), [call(m, "execute", &args)?])?.into_any())
}
#[pyfunction]
#[allow(clippy::too_many_arguments)] // Comfy node ABI; field names preserved for saved workflows.
fn prompt_generate_native(
    module: &Bound<'_, PyAny>,
    clip: &Bound<'_, PyAny>,
    prompt: &Bound<'_, PyAny>,
    system_prompt: &Bound<'_, PyAny>,
    media_strategy: &str,
    max_length: &Bound<'_, PyAny>,
    image: Option<&Bound<'_, PyAny>>,
    media: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let py = module.py();
    let text = module
        .getattr("build_generation_prompt")?
        .call1((prompt, system_prompt))?;
    let max = max_length
        .extract::<u16>()
        .ok()
        .filter(|n| (16..=2048).contains(n));
    if max.is_none() || max_length.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyValueError::new_err(
            "Remote Prompt Generate max_length must be 16 to 2048",
        ));
    }
    let videos = PyList::empty(py);
    for index in 0..3 {
        if let Some(video) = media.get_item(format!("video_{index}"))? {
            if !video.is_none() {
                videos.append(video)?;
            }
        }
    }
    let kwargs = PyDict::new(py);
    match media_strategy {
        "qwen-image" => {
            if !videos.is_empty() {
                return Err(PyValueError::new_err(
                    "Remote Prompt Generate qwen-image does not accept video",
                ));
            }
            kwargs.set_item("image", image)?;
            kwargs.set_item("skip_template", false)?;
            kwargs.set_item("min_length", 1)?;
            kwargs.set_item("thinking", false)?;
        }
        "minimax-ref2va" => {
            let refs = PyList::empty(py);
            if let Some(image) = image.filter(|v| !v.is_none()) {
                let count = image.getattr("shape")?.get_item(0)?.extract::<usize>()?;
                for i in 0..count {
                    let item = PyDict::new(py);
                    item.set_item("type", "image")?;
                    item.set_item(
                        "data",
                        image.get_item(pyo3::types::PySlice::new(
                            py,
                            i as isize,
                            i as isize + 1,
                            1,
                        ))?,
                    )?;
                    refs.append(item)?;
                }
            }
            for video in videos.iter() {
                let item = PyDict::new(py);
                item.set_item("type", "video")?;
                item.set_item("data", video)?;
                refs.append(item)?;
            }
            kwargs.set_item("minimax_ref_items", refs)?;
        }
        _ => {
            return Err(PyValueError::new_err(
                "Remote Prompt Generate media strategy is invalid",
            ));
        }
    }
    let tokens = clip.call_method("tokenize", (text,), Some(&kwargs))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("do_sample", false)?;
    kwargs.set_item("max_length", max.unwrap())?;
    let generated = clip.call_method("generate", (tokens,), Some(&kwargs))?;
    Ok(
        PyTuple::new(py, [clip.call_method1("decode", (generated,))?])?
            .into_any()
            .unbind(),
    )
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

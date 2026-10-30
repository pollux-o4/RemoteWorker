//! Media socket, model selection, cache identity and peer contract policy.
use crate::from_python;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use serde_json::Value;
fn invalid(s: impl Into<String>) -> PyErr {
    PyValueError::new_err(s.into())
}
#[pyfunction]
fn clean_model_name_native(
    value: &Bound<'_, PyAny>,
    prefix: &str,
    field_name: &str,
    strip: bool,
) -> PyResult<String> {
    let s = value
        .extract::<String>()
        .ok()
        .filter(|s| !s.trim().is_empty() && !s.contains('\0'))
        .ok_or_else(|| invalid(format!("{prefix} {field_name} must be a non-empty string")))?;
    Ok(if strip { s.trim().to_owned() } else { s })
}
#[pyfunction]
fn minimax_policy_native(value: &Bound<'_, PyAny>, cache: bool) -> PyResult<String> {
    let s = value.extract::<String>().unwrap_or_default();
    let allowed = if cache {
        ["reuse_same_inputs", "always_refresh"]
    } else {
        ["release_after_result", "keep"]
    };
    if !allowed.contains(&s.as_str()) {
        return Err(invalid(format!(
            "Remote MiniMax H3 {} must be one of: {}",
            if cache {
                "cache_policy"
            } else {
                "peer_memory_policy"
            },
            allowed.join(", ")
        )));
    }
    Ok(s)
}
#[pyfunction]
fn qwen_connection_native(
    server_url: &Bound<'_, PyAny>,
    timeout_seconds: &Bound<'_, PyAny>,
    remote_text_encoder: &Bound<'_, PyAny>,
    vae_name: &Bound<'_, PyAny>,
) -> PyResult<(String, i64, String, String)> {
    let url = clean_model_name_native(server_url, "Remote Qwen Image 2.1", "server_url", false)?;
    let timeout = timeout_seconds
        .extract::<i64>()
        .ok()
        .filter(|n| (5..=1800).contains(n));
    if timeout.is_none() || timeout_seconds.is_instance_of::<pyo3::types::PyBool>() {
        return Err(invalid(
            "Remote Qwen Image 2.1 timeout_seconds must be 5 to 1800",
        ));
    }
    Ok((
        url,
        timeout.unwrap(),
        clean_model_name_native(
            remote_text_encoder,
            "Remote Qwen Image 2.1",
            "remote_text_encoder",
            false,
        )?,
        clean_model_name_native(vae_name, "Remote Qwen Image 2.1", "vae_name", false)?,
    ))
}
fn connected<'py>(
    prefix: &str,
    media: &Bound<'py, PyDict>,
    maximum: usize,
    prompt: bool,
    start: usize,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let mut entries = Vec::new();
    for (key, value) in media.iter() {
        let name = key.extract::<String>()?;
        if let Some((head, tail)) = name.rsplit_once('_') {
            if head == prefix && !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(index) = tail.parse::<usize>() {
                    entries.push((index, name, value));
                }
            }
        }
    }
    entries.sort_by_key(|e| e.0);
    if prompt
        && (entries.len() > maximum || entries.iter().enumerate().any(|(i, e)| e.0 != i + start))
    {
        return Err(invalid(format!(
            "Remote Prompt Enhancer {prefix} sockets must be contiguous and contain at most {maximum} references"
        )));
    }
    if !prompt {
        if entries.len() > maximum {
            return Err(invalid(format!(
                "Remote MiniMax H3 accepts at most {maximum} {prefix} references"
            )));
        }
        if entries.iter().enumerate().any(|(i, e)| e.0 != i + start) {
            return Err(invalid(format!(
                "Remote MiniMax H3 {prefix} sockets must be connected without gaps"
            )));
        }
    }
    Ok(entries.into_iter().map(|(_, n, v)| (n, v)).collect())
}
#[pyfunction]
fn connected_slots_native(
    prefix: &str,
    media: &Bound<'_, PyDict>,
    maximum: usize,
) -> PyResult<Vec<(String, Py<PyAny>)>> {
    Ok(connected(prefix, media, maximum, false, 0)?
        .into_iter()
        .map(|(n, v)| (n, v.unbind()))
        .collect())
}
fn shape(image: &Bound<'_, PyAny>, name: &str, prompt: bool) -> PyResult<Vec<i64>> {
    let raw = image.getattr("shape").ok();
    let dims = raw.as_ref().and_then(|s| s.extract::<Vec<i64>>().ok());
    let error = if prompt {
        format!("Remote Prompt Enhancer {name} must be a BHWC IMAGE tensor")
    } else {
        format!("Remote MiniMax H3 {name} must be a BHWC IMAGE tensor")
    };
    let dims = dims
        .filter(|d| d.len() == 4)
        .ok_or_else(|| invalid(error))?;
    if dims[0] <= 0 || !matches!(dims[3], 3 | 4) || (!prompt && (dims[1] <= 0 || dims[2] <= 0)) {
        if prompt {
            return Err(invalid(format!(
                "Remote Prompt Enhancer {name} must be a BHWC IMAGE tensor"
            )));
        }
        let raw = raw.unwrap().repr()?.extract::<String>()?;
        return Err(invalid(format!(
            "Remote MiniMax H3 {name} has an invalid IMAGE shape: {raw}"
        )));
    }
    Ok(dims)
}
#[pyfunction]
fn ordered_images_native(
    prefix: &str,
    media: &Bound<'_, PyDict>,
    batches: bool,
    maximum: usize,
    prompt: bool,
) -> PyResult<Vec<Py<PyAny>>> {
    let mut result = Vec::new();
    for (name, image) in connected(prefix, media, maximum, prompt, 0)? {
        if image.is_none() && !prompt {
            return Err(invalid(format!(
                "Remote MiniMax H3 requires connected {name}"
            )));
        }
        let dims = shape(&image, if prompt { prefix } else { &name }, prompt)?;
        if !prompt {
            if !batches && dims[0] != 1 {
                return Err(invalid(format!(
                    "Remote MiniMax H3 {name} must contain exactly one reference image"
                )));
            }
            if batches && dims[0] < 5 {
                return Err(invalid(format!(
                    "Remote MiniMax H3 {name} needs at least 5 frames"
                )));
            }
        }
        result.push(image.unbind());
    }
    Ok(result)
}
#[pyfunction]
fn ordered_audio_native(
    prefix: &str,
    media: &Bound<'_, PyDict>,
    maximum: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let mut result = Vec::new();
    for (name, audio) in connected(prefix, media, maximum, false, 0)? {
        if !audio.is_instance_of::<PyDict>()
            || !audio.contains("waveform")?
            || !audio.contains("sample_rate")?
        {
            return Err(invalid(format!(
                "Remote MiniMax H3 {name} must be a connected AUDIO value"
            )));
        }
        result.push(audio.unbind());
    }
    Ok(result)
}
#[pyfunction]
fn single_frame_native(image: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    shape(image, name, false)?;
    Ok(image.get_item(0)?.unbind())
}
#[pyfunction]
fn qwen_socket_count_native(images: &Bound<'_, PyDict>) -> PyResult<usize> {
    let mut indices = Vec::new();
    for (key, _) in images.iter() {
        let key = key.extract::<String>()?;
        if let Some(s) = key.strip_prefix("image_") {
            if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(i) = s.parse::<usize>() {
                    indices.push(i);
                }
            }
        }
    }
    indices.sort();
    if indices.len() > 16 || indices.iter().enumerate().any(|(i, n)| *n != i + 1) {
        return Err(invalid(
            "Remote Qwen Image 2.1 accepts up to 16 consecutive image sockets",
        ));
    }
    Ok(indices.len())
}
#[pyfunction]
fn validate_qwen_frame_count_native(count: usize) -> PyResult<()> {
    if count > 16 {
        return Err(invalid(
            "Remote Qwen Image 2.1 accepts at most 16 image attachments",
        ));
    }
    Ok(())
}
#[pyfunction]
pub(crate) fn qwen_worker_inputs_native(
    prompt: &Bound<'_, PyAny>,
    negative_prompt: &Bound<'_, PyAny>,
    resolution: &Bound<'_, PyAny>,
) -> PyResult<i64> {
    if prompt
        .extract::<String>()
        .ok()
        .is_none_or(|s| s.trim().is_empty())
    {
        return Err(invalid("Remote Qwen Image 2.1 requires a non-empty prompt"));
    }
    if !negative_prompt.is_instance_of::<pyo3::types::PyString>() {
        return Err(invalid(
            "Remote Qwen Image 2.1 negative_prompt must be a string",
        ));
    }
    if resolution.is_instance_of::<pyo3::types::PyBool>() {
        return Err(invalid(
            "Remote Qwen Image 2.1 resolution must be 0 to 4096",
        ));
    }
    resolution
        .extract::<i64>()
        .ok()
        .filter(|n| (0..=4096).contains(n))
        .ok_or_else(|| invalid("Remote Qwen Image 2.1 resolution must be 0 to 4096"))
}
#[pyfunction]
pub(crate) fn qwen_resize_native(width: i64, height: i64, resolution: i64) -> PyResult<(i64, i64)> {
    if width <= 0 || height <= 0 {
        return Err(invalid(
            "Remote Qwen Image 2.1 image dimensions must be positive",
        ));
    }
    let (w, h) = if resolution > 0 {
        let ratio = width as f64 / height as f64;
        (
            ((resolution as f64).powi(2) * ratio).sqrt(),
            ((resolution as f64).powi(2) / ratio).sqrt(),
        )
    } else {
        (width as f64, height as f64)
    };
    Ok((
        (((w / 32.0).round_ties_even() as i64) * 32).max(32),
        (((h / 32.0).round_ties_even() as i64) * 32).max(32),
    ))
}
#[pyfunction]
fn validate_prompt_media_native(
    profile: &Bound<'_, PyAny>,
    images: usize,
    videos: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let count = videos.len()?;
    if images > 8 || count > 3 {
        return Err(invalid(
            "Remote Prompt Enhancer media exceeds the request limit",
        ));
    }
    for video in videos.try_iter()? {
        let video = video?;
        let shape = video.getattr("shape").ok();
        let frames = shape
            .and_then(|s| s.get_item(0).ok())
            .and_then(|s| s.extract::<i64>().ok())
            .unwrap_or(0);
        if frames > 32 {
            return Err(invalid(
                "Remote Prompt Enhancer video exceeds the frame limit",
            ));
        }
    }
    let accepted = profile.getattr("accepted_media")?;
    let id: String = profile.getattr("id")?.extract()?;
    if images > 0 && !accepted.contains("image")? {
        return Err(invalid(format!(
            "Remote Prompt Enhancer profile {id} does not accept images"
        )));
    }
    if count > 0 && !accepted.contains("video")? {
        return Err(invalid(format!(
            "Remote Prompt Enhancer profile {id} does not accept video"
        )));
    }
    Ok(())
}
fn node_id(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}
fn connected_vae(workflow: &Value, id: &str, input_name: &str) -> Option<Value> {
    let nodes = workflow.get("nodes")?.as_array()?;
    let current = nodes
        .iter()
        .find(|n| n.get("id").is_some_and(|v| node_id(v) == id))?;
    let input = current
        .get("inputs")?
        .as_array()?
        .iter()
        .find(|field| field["name"] == input_name && !field["link"].is_null())?;
    let link_id = node_id(&input["link"]);
    let mut origin = None;
    for link in workflow.get("links")?.as_array()? {
        if link.is_object() && node_id(&link["id"]) == link_id {
            origin = link.get("origin_id").cloned();
        } else if let Some(items) = link.as_array() {
            if items.len() >= 2 && node_id(&items[0]) == link_id {
                origin = Some(items[1].clone());
            }
        }
        if origin.is_some() {
            break;
        }
    }
    let origin = origin?;
    let source = nodes
        .iter()
        .find(|n| n.get("id").is_some_and(|v| node_id(v) == node_id(&origin)))?;
    let kind = source
        .get("type")
        .or_else(|| source.get("class_type"))
        .or_else(|| source.get("title"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if !kind.contains("VAELoader") && !kind.contains("VAE 로드") {
        return None;
    }
    source.pointer("/widgets_values/0").cloned()
}
#[pyfunction]
#[pyo3(signature=(widget,connected,input_name,unique_id,extra))]
fn resolve_vae_name_native(
    widget: &Bound<'_, PyAny>,
    connected: Option<&Bound<'_, PyAny>>,
    input_name: Option<&str>,
    unique_id: Option<&Bound<'_, PyAny>>,
    extra: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let candidate = connected.filter(|v| !v.is_none()).unwrap_or(widget);
    if candidate.is_instance_of::<pyo3::types::PyString>() {
        return clean_model_name_native(candidate, "Remote MiniMax H3", "VAE name", true);
    }
    if let (Some(input), Some(id)) = (input_name, unique_id.filter(|id| !id.is_none())) {
        let id: String = id.str()?.extract()?;
        if let Ok(extra) = from_python(extra) {
            let workflow = extra.get("workflow").filter(|v| v.is_object()).or_else(|| {
                extra
                    .pointer("/extra_pnginfo/workflow")
                    .filter(|v| v.is_object())
            });
            if let Some(name) = workflow.and_then(|w| connected_vae(w, &id, input)) {
                let name = crate::to_python(candidate.py(), &name)?;
                return clean_model_name_native(
                    name.bind(candidate.py()),
                    "Remote MiniMax H3",
                    "VAE name",
                    true,
                );
            }
        }
    }
    Err(invalid(
        "Remote MiniMax H3 VAE override must be a VAE name string or a directly connected VAELoader; use Remote VAE Name Selector for peer-only loading",
    ))
}
#[pyfunction]
fn add_release_cleanup_native(
    workflow: &Bound<'_, PyDict>,
    policy: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let policy = minimax_policy_native(policy, false)?;
    if policy != "release_after_result" {
        return Ok(());
    }
    let py = workflow.py();
    let inputs = workflow
        .get_item("save_latent")?
        .and_then(|v| v.get_item("inputs").ok())
        .ok_or_else(|| {
            PyRuntimeError::new_err(
                "Remote MiniMax H3 workflow is missing SaveRemoteLatent for cleanup ordering",
            )
        })?;
    inputs.set_item("dependency", ("save", 0).into_pyobject(py)?.to_list())?;
    let anything = PyList::new(
        py,
        [
            "save_latent".into_pyobject(py)?.into_any(),
            0i32.into_pyobject(py)?.into_any(),
        ],
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("anything", anything)?;
    kwargs.set_item("offload_model", true)?;
    kwargs.set_item("offload_cache", true)?;
    kwargs.set_item("reset_execution_cache", false)?;
    let node = PyDict::new(py);
    node.set_item("class_type", "VRAMCleanup")?;
    node.set_item("inputs", kwargs)?;
    workflow.set_item("cleanup", node)?;
    Ok(())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clean_model_name_native, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_policy_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_connection_native, m)?)?;
    m.add_function(wrap_pyfunction!(connected_slots_native, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_images_native, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_audio_native, m)?)?;
    m.add_function(wrap_pyfunction!(single_frame_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_socket_count_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_qwen_frame_count_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_worker_inputs_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_resize_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_prompt_media_native, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_vae_name_native, m)?)?;
    m.add_function(wrap_pyfunction!(add_release_cleanup_native, m)?)?;
    m.add_function(wrap_pyfunction!(cache_identity_native, m)?)?;
    m.add_function(wrap_pyfunction!(cache_fingerprint_native, m)?)?;
    m.add_function(wrap_pyfunction!(encode_audio_wav_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_latent_artifact_native, m)?)?;
    m.add_function(wrap_pyfunction!(peer_loaders_native, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_cached_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_preparation_inputs_native, m)?)?;
    m.add_function(wrap_pyfunction!(encoder_local_inputs_native, m)?)?;
    m.add_function(wrap_pyfunction!(release_capability_native, m)?)?;
    Ok(())
}

#[pyfunction]
fn cache_identity_native(kind: &str, values: &Bound<'_, PyDict>) -> PyResult<Py<PyTuple>> {
    let py = values.py();
    let get = |key: &str| {
        values
            .get_item(key)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))
    };
    let fields: &[&str] = match kind {
        "conditioning" => &[
            "strategy_name",
            "clip",
            "prompt",
            "frames",
            "grounding_px",
            "system_prompt",
            "server_url",
        ],
        "qwen" => &[
            "clip",
            "vae_name",
            "prompt",
            "negative_prompt",
            "resolution",
            "negative_vision",
            "frames",
            "server_url",
        ],
        "minimax-i2v" => &[
            "clip",
            "vae_name",
            "prompt",
            "width",
            "height",
            "length",
            "first_frame",
            "last_frame",
            "server_url",
        ],
        "minimax-reference" => &[
            "clip",
            "vae_name",
            "audio_vae_name",
            "prompt",
            "width",
            "height",
            "length",
            "ref_image_size",
            "images",
            "videos",
            "video_audios",
            "audios",
            "server_url",
        ],
        _ => return Err(invalid("Unknown remote cache kind")),
    };
    let mut output = Vec::new();
    if kind != "conditioning" {
        output.push(
            match kind {
                "qwen" => "qwen-image-2.1",
                "minimax-i2v" => "minimax-h3-i2v",
                _ => "minimax-h3-ref2va",
            }
            .into_pyobject(py)?
            .into_any()
            .unbind(),
        );
    }
    for field in fields {
        let value = get(field)?;
        let value = if *field == "server_url" {
            value
                .str()?
                .extract::<String>()?
                .trim_end_matches('/')
                .into_pyobject(py)?
                .into_any()
                .unbind()
        } else if matches!(*field, "width" | "height" | "length") {
            py.import("builtins")?
                .getattr("int")?
                .call1((value,))?
                .unbind()
        } else if matches!(
            *field,
            "frames" | "images" | "videos" | "video_audios" | "audios"
        ) {
            PyTuple::new(py, value.try_iter()?.collect::<PyResult<Vec<_>>>()?)?
                .into_any()
                .unbind()
        } else {
            value.unbind()
        };
        output.push(value);
    }
    Ok(PyTuple::new(py, output)?.unbind())
}
#[pyfunction]
#[pyo3(signature=(cache_policy,prefix="Remote Conditioning"))]
fn cache_fingerprint_native(cache_policy: &Bound<'_, PyAny>, prefix: &str) -> PyResult<f64> {
    let policy = cache_policy.extract::<String>().unwrap_or_default();
    match policy.as_str() {
        "always_refresh" => Ok(f64::NAN),
        "reuse_same_inputs" => Ok(0.0),
        _ => Err(invalid(format!(
            "{prefix} cache_policy must be one of: reuse_same_inputs, always_refresh"
        ))),
    }
}

#[pyfunction]
fn encode_audio_wav_native(
    sample_rate: &Bound<'_, PyAny>,
    shape: Vec<usize>,
    data: Vec<u8>,
) -> PyResult<Vec<u8>> {
    let rate = sample_rate
        .extract::<u32>()
        .ok()
        .filter(|n| (1..=384000).contains(n))
        .ok_or_else(|| invalid("Remote MiniMax H3 audio sample_rate is invalid"))?;
    if shape.len() != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] == 0 {
        return Err(invalid(
            "Remote MiniMax H3 AUDIO waveform must have shape [1, channels, samples]",
        ));
    }
    let channels = shape[1];
    let samples = shape[2];
    let expected = channels
        .checked_mul(samples)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("Remote MiniMax H3 AUDIO waveform is too large"))?;
    if data.len() != expected {
        return Err(invalid(
            "Remote MiniMax H3 AUDIO waveform data is malformed",
        ));
    }
    let mut values = Vec::with_capacity(channels * samples);
    for bytes in data.chunks_exact(4) {
        let value = f32::from_le_bytes(bytes.try_into().unwrap());
        if !value.is_finite() {
            return Err(invalid(
                "Remote MiniMax H3 AUDIO waveform contains non-finite values",
            ));
        }
        values.push((value.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16);
    }
    let size = channels
        .checked_mul(samples)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n <= u32::MAX - 36)
        .ok_or_else(|| invalid("Remote MiniMax H3 AUDIO waveform is too large"))?;
    let channel_count = u16::try_from(channels)
        .ok()
        .filter(|c| *c <= u16::MAX / 2)
        .ok_or_else(|| invalid("Remote MiniMax H3 AUDIO channel count is invalid"))?;
    let align = channel_count * 2;
    let byte_rate = rate
        .checked_mul(align as u32)
        .ok_or_else(|| invalid("Remote MiniMax H3 AUDIO channel count is invalid"))?;
    let mut wav = Vec::with_capacity(44 + size as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(size + 36).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channel_count.to_le_bytes());
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&size.to_le_bytes());
    for i in 0..samples {
        for c in 0..channels {
            wav.extend_from_slice(&values[c * samples + i].to_le_bytes());
        }
    }
    Ok(wav)
}
#[pyfunction]
fn validate_latent_artifact_native(state: &Bound<'_, PyAny>, prefix: &str) -> PyResult<()> {
    if !state.is_instance_of::<PyDict>() || !state.contains("samples")? {
        return Err(PyRuntimeError::new_err(format!(
            "{prefix} latent artifact is invalid"
        )));
    }
    Ok(())
}
#[pyfunction]
fn peer_loaders_native(
    kind: &str,
    module: &Bound<'_, PyAny>,
    api: &Bound<'_, PyAny>,
    clip: &Bound<'_, PyAny>,
    vae_names: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let py = module.py();
    let qwen = kind == "qwen";
    let prefix = if qwen {
        "Remote Qwen Image 2.1"
    } else {
        "Remote MiniMax H3"
    };
    let options = api.call_method0("clip_loader_options")?;
    let clip = if qwen {
        let choices = options.call_method1("get", ("clip_name", PyList::empty(py)))?;
        let resolved = module.getattr("resolve_peer_model_name")?.call1((
            clip.getattr("clip_name")?,
            choices,
            "clip_name",
        ))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("clip_name", resolved)?;
        clip.call_method("_replace", (), Some(&kwargs))?
    } else {
        clip.clone()
    };
    for (i, name) in ["clip_name", "clip_type", "device"].iter().enumerate() {
        let value = clip.get_item(i)?;
        if !options
            .call_method1("get", (*name, PyList::empty(py)))?
            .contains(&value)?
        {
            return Err(invalid(format!(
                "{prefix} peer does not provide {name}: {}",
                value.repr()?.to_str()?
            )));
        }
    }
    let info = api.call_method1("require_node", ("VAELoader",))?;
    let parsed = (|| -> PyResult<Bound<'_, PyAny>> {
        info.get_item("input")?
            .get_item("required")?
            .get_item("vae_name")?
            .get_item(0)
    })();
    let options = parsed.map_err(|_| {
        PyRuntimeError::new_err(format!("{prefix} peer VAELoader capability is malformed"))
    })?;
    if qwen {
        if !options.is_instance_of::<PyList>() {
            return Err(PyRuntimeError::new_err(
                "Remote Qwen Image 2.1 peer VAELoader capability is malformed",
            ));
        }
        let vae = module.getattr("resolve_peer_model_name")?.call1((
            vae_names
                .first()
                .ok_or_else(|| invalid("Missing Qwen VAE name"))?,
            options,
            "vae_name",
        ))?;
        return Ok(PyTuple::new(py, [clip, vae])?.into_any().unbind());
    }
    if !options.is_instance_of::<PyList>() {
        return Err(invalid(
            "Remote MiniMax H3 peer does not provide the selected VAE",
        ));
    }
    for name in vae_names {
        if !options.contains(name.bind(py))? {
            return Err(invalid(
                "Remote MiniMax H3 peer does not provide the selected VAE",
            ));
        }
    }
    Ok(py.None())
}
#[pyfunction]
fn minimax_cached_native(cache: &Bound<'_, PyAny>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let value = cache.call_method1("get", (key,))?;
    if value.is_none() {
        return Ok(value.unbind());
    }
    if value.len()? != 2 {
        cache.call_method0("clear")?;
        return Ok(cache.py().None());
    }
    Ok(value.unbind())
}
#[pyfunction]
fn qwen_preparation_inputs_native(values: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(values.py());
    for (key, value) in values.iter() {
        let key = key.extract::<String>()?;
        result.set_item(key.strip_prefix("images.").unwrap_or(&key), value)?;
    }
    Ok(result.unbind())
}

#[pyfunction]
fn encoder_local_inputs_native(kind: &str, values: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
    let py = values.py();
    let get = |key: &str| {
        values
            .get_item(key)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))
    };
    let result = (|| -> PyResult<()> {
        match kind {
            "conditioning" => {
                for name in ["remote_text_encoder", "clip_type", "device"] {
                    let value = get(name)?;
                    if value
                        .extract::<String>()
                        .ok()
                        .is_none_or(|s| s.trim().is_empty())
                    {
                        return Err(invalid(format!(
                            "Remote Conditioning {name} must be a non-empty string"
                        )));
                    }
                }
                for (name, allowed) in [
                    ("peer_memory_policy", ["release_after_result", "keep"]),
                    ("cache_policy", ["reuse_same_inputs", "always_refresh"]),
                ] {
                    let value = get(name)?.extract::<String>().unwrap_or_default();
                    if !allowed.contains(&value.as_str()) {
                        return Err(invalid(format!(
                            "Remote Conditioning {name} must be one of: {}",
                            allowed.join(", ")
                        )));
                    }
                }
            }
            "qwen" => {
                for name in ["remote_text_encoder", "vae_name"] {
                    clean_model_name_native(&get(name)?, "Remote Qwen Image 2.1", name, false)?;
                }
            }
            "minimax" => {
                for name in ["clip_name", "clip_type", "device"] {
                    clean_model_name_native(&get(name)?, "Remote MiniMax H3", name, true)?;
                }
                for vae in get("vae_names")?.try_iter()? {
                    clean_model_name_native(&vae?, "Remote MiniMax H3", "VAE name", true)?;
                }
                minimax_policy_native(&get("peer_memory_policy")?, false)?;
                minimax_policy_native(&get("cache_policy")?, true)?;
            }
            "prompt" => {
                let value = get("profile_id")?;
                if crate::profiles::validate_profile_id_native(&value).is_err() {
                    return Err(invalid("Remote Prompt Enhancer profile_id is invalid"));
                }
            }
            _ => return Err(invalid("Unknown encoder validation kind")),
        }
        Ok(())
    })();
    match result {
        Ok(()) => Ok(true.into_pyobject(py)?.to_owned().into_any().unbind()),
        Err(error) => Ok(error.value(py).str()?.into_any().unbind()),
    }
}

#[pyfunction]
fn release_capability_native(
    api: &Bound<'_, PyAny>,
    policy: &Bound<'_, PyAny>,
    validate: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let policy = minimax_policy_native(policy, false)?;
    if policy == "release_after_result" {
        validate.call1((api.call_method1("require_node", ("VRAMCleanup",))?,))?;
    }
    Ok(())
}

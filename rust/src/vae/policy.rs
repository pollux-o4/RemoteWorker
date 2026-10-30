use crate::{from_python, to_python};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyFloat, PyInt, PyString},
};
use serde_json::{Value, json};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Image,
    Audio,
}
impl Kind {
    fn parse(value: &str, message: &str) -> PyResult<Self> {
        match value {
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            _ => Err(PyValueError::new_err(message.to_owned())),
        }
    }
}
fn model(value: &Bound<'_, PyAny>) -> bool {
    value
        .extract::<String>()
        .is_ok_and(|s| !s.trim().is_empty() && !s.contains('\0'))
}
#[pyfunction]
fn vae_validate_model_choice(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if model(value) {
        Ok(true
            .into_pyobject(value.py())?
            .to_owned()
            .into_any()
            .unbind())
    } else {
        Ok(
            "원격 VAE를 선택하세요. 노드의 모델 목록 새로고침으로 목록 조회 가능."
                .into_pyobject(value.py())?
                .into_any()
                .unbind(),
        )
    }
}
#[pyfunction]
#[pyo3(signature=(vae_name,audio_vae_name,fps,crf,tiled,kind="image"))]
fn vae_video_options(
    vae_name: &Bound<'_, PyAny>,
    audio_vae_name: &Bound<'_, PyAny>,
    fps: &Bound<'_, PyAny>,
    crf: &Bound<'_, PyAny>,
    tiled: &Bound<'_, PyAny>,
    kind: &str,
) -> PyResult<Py<PyDict>> {
    let media = Kind::parse(kind, "Unknown shared media kind")?;
    if !model(vae_name) {
        return Err(PyValueError::new_err("영상 VAE를 선택하세요."));
    }
    if !audio_vae_name.is_instance_of::<PyString>()
        || audio_vae_name
            .extract::<String>()
            .is_ok_and(|s| s.contains('\0'))
    {
        return Err(PyValueError::new_err("Invalid audio VAE name"));
    }
    let py = vae_name.py();
    let fps_number = fps.extract::<f64>().ok();
    let fps_type = fps.get_type();
    let crf_type = crf.get_type();
    if !(fps_type.is(py.get_type::<PyInt>()) || fps_type.is(py.get_type::<PyFloat>()))
        || fps_number.is_none_or(|v| !v.is_finite() || !(1.0..=120.0).contains(&v))
    {
        return Err(PyValueError::new_err(
            "Video fps must be between 1.0 and 120.0",
        ));
    }
    if !crf_type.is(py.get_type::<PyInt>())
        || crf
            .extract::<i64>()
            .ok()
            .is_none_or(|v| !(0..=51).contains(&v))
    {
        return Err(PyValueError::new_err("Video crf must be between 0 and 51"));
    }
    let result = PyDict::new(py);
    for (k, v) in [
        ("vae_name", vae_name),
        ("audio_vae_name", audio_vae_name),
        ("fps", fps),
        ("crf", crf),
    ] {
        result.set_item(k, v)?;
    }
    result.set_item("tiled", tiled.is_truthy()?)?;
    if media == Kind::Audio {
        result.set_item("kind", kind)?;
    }
    Ok(result.unbind())
}
fn cleanup(graph: &mut Value, node: &str) {
    graph["cleanup"] = json!({"class_type":"VRAMCleanup","inputs":{"anything":[node,0],"offload_model":true,"offload_cache":true,"reset_execution_cache":false}});
}
#[pyfunction]
pub(super) fn vae_build_workflow(
    py: Python<'_>,
    vae_name: &str,
    filename: &str,
    kind: &str,
    options: &Bound<'_, PyAny>,
    request_id: &str,
    memory_policy: &str,
) -> PyResult<Py<PyAny>> {
    let mut decode = from_python(options)?
        .as_object()
        .cloned()
        .ok_or_else(|| PyValueError::new_err("Invalid decode options"))?;
    // Caller order matches the public adapter: explicit options may override native fields.
    let mut fields = json!({"vae":["vae",0],"filename":filename,"kind":kind})
        .as_object()
        .unwrap()
        .clone();
    fields.append(&mut decode);
    let mut graph = json!({"vae":{"class_type":"VAELoader","inputs":{"vae_name":vae_name}},"decode":{"class_type":"RemoteVAEDecodeWorker","inputs":fields},"save":{"class_type":"SaveRemoteVAEOutput","inputs":{"value":["decode",0],"filename_prefix":format!("remote-conditioning/{request_id}/decoded")}}});
    if memory_policy == "release_after_result" {
        cleanup(&mut graph, "save");
    }
    to_python(py, &graph)
}
#[pyfunction]
#[pyo3(signature=(api,vae_name,worker,memory_policy="keep"))]
pub(super) fn vae_preflight(
    api: &Bound<'_, PyAny>,
    vae_name: &Bound<'_, PyAny>,
    worker: &str,
    memory_policy: &str,
) -> PyResult<()> {
    if !model(vae_name) {
        return Err(PyValueError::new_err("Remote VAE requires a model name"));
    }
    if !matches!(memory_policy, "keep" | "release_after_result") {
        return Err(PyValueError::new_err("Unknown Remote VAE memory policy"));
    }
    api.call_method1("require_node", (worker,))?;
    let info = from_python(&api.call_method1("require_node", ("VAELoader",))?)?;
    let name = vae_name.extract::<String>()?;
    if !info
        .pointer("/input/required/vae_name/0")
        .and_then(Value::as_array)
        .is_some_and(|a| a.iter().any(|s| s.as_str() == Some(&name)))
    {
        return Err(PyValueError::new_err(format!(
            "처리 서버에 선택한 VAE가 없습니다: {name}"
        )));
    }
    if memory_policy == "release_after_result" {
        let info = api.call_method1("require_node", ("VRAMCleanup",))?;
        crate::protocol::validate_cleanup_capability(&info)?;
    }
    Ok(())
}
#[pyfunction]
fn vae_warmup_shape(
    channels: usize,
    dims: usize,
    output_channels: Option<usize>,
    kind: &str,
) -> PyResult<Vec<usize>> {
    let shape = match (kind, dims) {
        ("audio", 2) if channels == 32 => vec![1, channels, 2, 8],
        ("image", 3) => vec![1, channels, 2, 2, 2],
        ("image", 2) if output_channels.unwrap_or(3) != 2 => vec![1, channels, 8, 8],
        _ => {
            return Err(PyValueError::new_err(
                "Dummy warmup is unsupported for this VAE family",
            ));
        }
    };
    Ok(shape)
}
pub(super) fn vae_check_channels(kind: &str, channels: Option<usize>) -> PyResult<()> {
    if kind == "audio" && channels.is_some_and(|v| v >= 3) {
        return Err(PyValueError::new_err(
            "선택한 모델은 이미지·영상 VAE입니다. 오디오 VAE를 선택하세요.",
        ));
    }
    if kind == "image" && channels == Some(2) {
        return Err(PyValueError::new_err(
            "선택한 모델은 오디오 VAE입니다. 오디오 디코딩 또는 오디오 대상을 선택하세요.",
        ));
    }
    Ok(())
}
pub(super) fn vae_check_audio_shape(batch: usize, channels: usize) -> PyResult<()> {
    if batch != 1 {
        return Err(PyValueError::new_err(
            "공유 오디오 저장은 배치 1만 지원합니다.",
        ));
    }
    if !matches!(channels, 1 | 2) {
        return Err(PyValueError::new_err(
            "공유 WAV 저장은 모노·스테레오만 지원합니다.",
        ));
    }
    Ok(())
}
#[pyfunction]
fn vae_check_budget(elements: usize, max_bytes: usize) -> PyResult<()> {
    if elements.saturating_mul(4) > max_bytes.saturating_sub(1024 * 1024) {
        return Err(PyValueError::new_err(
            "Decoded output exceeds the Remote VAE transfer budget; reduce the batch or clip length",
        ));
    }
    Ok(())
}
#[pyfunction]
pub(super) fn vae_tensor_options(
    py: Python<'_>,
    vae_name: &Bound<'_, PyAny>,
    kind: &str,
    decode_options: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    Kind::parse(kind, "Unknown Remote VAE output kind")?;
    let mut options = decode_defaults();
    options["delivery"] = "tensor".into();
    options["vae_name"] = from_python(vae_name)?;
    options["kind"] = kind.into();
    for (k, v) in decode_options.iter() {
        let k = k.extract::<String>()?;
        if !matches!(
            k.as_str(),
            "tiled" | "tile_size" | "overlap" | "temporal_size" | "temporal_overlap"
        ) {
            return Err(PyValueError::new_err("Unknown shared tensor decode option"));
        }
        options[k] = from_python(&v)?;
    }
    to_python(py, &options)
}
pub(super) fn vae_validate_kind(kind: &str, warmup: bool) -> PyResult<()> {
    Kind::parse(
        kind,
        if warmup {
            "Unknown Remote VAE warmup kind"
        } else {
            "Unknown Remote VAE output kind"
        },
    )?;
    Ok(())
}
pub(super) fn vae_warmup_graph(
    py: Python<'_>,
    vae_name: &str,
    kind: &str,
    request_id: &str,
) -> PyResult<Py<PyAny>> {
    to_python(
        py,
        &json!({"vae":{"class_type":"VAELoader","inputs":{"vae_name":vae_name}},"warmup":{"class_type":"RemoteVAEWarmupWorker","inputs":{"vae":["vae",0],"kind":kind}},"save":{"class_type":"SaveRemoteText","inputs":{"text":["warmup",0],"filename_prefix":format!("remote-conditioning/{request_id}/warmup")}}}),
    )
}
pub(super) fn vae_validate_ticket(
    py: Python<'_>,
    ticket: &Bound<'_, PyAny>,
    vae_name: &str,
    kind: &str,
    server_url: &str,
) -> PyResult<Option<Py<PyAny>>> {
    if ticket.is_none() || ticket.extract::<String>().is_ok_and(|s| s == "ready") {
        return Ok(None);
    }
    let raw = ticket
        .extract::<String>()
        .map_err(|_| PyValueError::new_err("Invalid Remote VAE warmup ticket"))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|_| PyValueError::new_err("Invalid Remote VAE warmup ticket"))?;
    let fields = ["prompt_id", "request_id", "server_url", "vae_name", "kind"];
    let valid = value.as_object().is_some_and(|o| {
        o.len() == 5
            && fields.iter().all(|k| {
                o.get(*k)
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty())
            })
    }) && value["server_url"] == server_url.trim_end_matches('/')
        && value["vae_name"] == vae_name
        && value["kind"] == kind;
    if !valid {
        return Err(PyValueError::new_err(
            "준비 신호와 처리 서버·VAE·대상이 다릅니다. 두 노드의 설정을 맞추세요.",
        ));
    }
    Ok(Some(to_python(py, &value)?))
}
pub(super) fn vae_ticket(
    py: Python<'_>,
    prompt_id: &Bound<'_, PyAny>,
    request_id: &str,
    server_url: &str,
    vae_name: &str,
    kind: &str,
) -> PyResult<String> {
    let id = prompt_id
        .extract::<String>()
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| PyRuntimeError::new_err("Remote VAE warmup returned no prompt_id"))?;
    let value = to_python(
        py,
        &json!({"prompt_id":id,"request_id":request_id,"server_url":server_url.trim_end_matches('/'),"vae_name":vae_name,"kind":kind}),
    )?;
    py.import("json")?
        .call_method1("dumps", (value,))?
        .extract()
}
#[pyfunction]
fn vae_validate_retry(latent: Option<String>, request_id: &str) -> PyResult<String> {
    if let Some(latent) = latent.filter(|s| !s.is_empty()) {
        if !request_id.is_empty() && latent != request_id {
            return Err(PyValueError::new_err(
                "연결된 latent와 재시도 작업 ID가 다릅니다.",
            ));
        }
        Ok(latent)
    } else {
        Ok(request_id.into())
    }
}
pub(super) fn vae_shared_workflow(
    py: Python<'_>,
    request_id: &str,
    options: &Bound<'_, PyAny>,
    memory_policy: &str,
    tensor: bool,
) -> PyResult<Py<PyAny>> {
    let mut settings = from_python(options)?
        .as_object()
        .cloned()
        .ok_or_else(|| PyValueError::new_err("Invalid shared VAE settings"))?;
    if tensor {
        settings.remove("delivery");
    }
    settings.insert("request_id".into(), request_id.into());
    let mut graph = json!({"save":{"class_type":if tensor{"SharedVAETensorWorker"}else{"SharedVAEVideoWorker"},"inputs":settings}});
    if memory_policy == "release_after_result" {
        cleanup(&mut graph, "save");
    }
    to_python(py, &graph)
}
pub(super) fn vae_extract_receipt(outputs: &Bound<'_, PyAny>, tensor: bool) -> PyResult<Py<PyAny>> {
    let value = from_python(outputs)?;
    let field = if tensor {
        "shared_tensor"
    } else {
        "shared_video"
    };
    let values = value
        .get("save")
        .and_then(|v| v.get(field))
        .and_then(Value::as_array)
        .filter(|v| v.len() == 1)
        .ok_or_else(|| {
            PyValueError::new_err(if tensor {
                "Shared VAE tensor receipt is missing"
            } else {
                "Shared VAE video receipt is missing"
            })
        })?;
    to_python(outputs.py(), &values[0])
}

#[pyfunction]
fn vae_lazy_status(
    py: Python<'_>,
    samples: &Bound<'_, PyAny>,
    request_id: &str,
    output_mode: &str,
) -> PyResult<Py<PyAny>> {
    to_python(
        py,
        &if output_mode == "shared" && !request_id.is_empty() {
            json!([])
        } else if samples.is_none() {
            json!(["samples"])
        } else {
            json!([])
        },
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VAE_VIDEO_FPS", 24.0)?;
    m.add("VAE_VIDEO_CRF", 18)?;
    m.add_function(wrap_pyfunction!(vae_video_fields, m)?)?;
    m.add_function(wrap_pyfunction!(vae_decode_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(vae_exclusive_sources, m)?)?;
    m.add_function(wrap_pyfunction!(vae_request_body, m)?)?;
    m.add_function(wrap_pyfunction!(vae_decode_validation, m)?)?;
    m.add_function(wrap_pyfunction!(vae_validate_peer_url, m)?)?;
    m.add_function(wrap_pyfunction!(vae_output_budget, m)?)?;
    m.add_function(wrap_pyfunction!(vae_upload_relative, m)?)?;
    m.add_function(wrap_pyfunction!(vae_validate_model_choice, m)?)?;
    m.add_function(wrap_pyfunction!(vae_video_options, m)?)?;
    m.add_function(wrap_pyfunction!(vae_build_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(vae_preflight, m)?)?;
    m.add_function(wrap_pyfunction!(vae_warmup_shape, m)?)?;
    m.add_function(wrap_pyfunction!(vae_tensor_options, m)?)?;
    m.add_function(wrap_pyfunction!(vae_validate_retry, m)?)?;
    m.add_function(wrap_pyfunction!(vae_lazy_status, m)?)?;
    Ok(())
}

#[pyfunction]
fn vae_validate_peer_url(
    server_url: &str,
    port: Option<u16>,
    own_hosts: Vec<String>,
) -> PyResult<()> {
    let parsed = url::Url::parse(server_url).map_err(|_| {
        PyValueError::new_err("Remote Conditioning server_url must be an absolute HTTP(S) origin")
    })?;
    if port.is_some()
        && parsed.port_or_known_default() == port
        && parsed
            .host_str()
            .is_some_and(|h| own_hosts.iter().any(|own| own == h))
    {
        return Err(PyValueError::new_err(
            "처리 서버가 현재 서버와 같습니다. 다른 ComfyUI 포트를 지정하세요.",
        ));
    }
    Ok(())
}
#[pyfunction]
pub(super) fn vae_output_budget(
    vae: &Bound<'_, PyAny>,
    samples: &Bound<'_, PyAny>,
    kind: &str,
    max_bytes: usize,
) -> PyResult<()> {
    let mut latent = samples.get_item("samples")?;
    if latent
        .getattr("is_nested")
        .map(|v| v.is_truthy())
        .unwrap_or(Ok(false))?
    {
        latent = latent
            .call_method0("unbind")?
            .get_item(if kind == "image" { 0 } else { -1 })?;
    }
    let shape: Vec<usize> = latent.getattr("shape")?.extract()?;
    let ratios = vae.getattr("upscale_ratio")?;
    let channels = vae
        .getattr("output_channels")
        .and_then(|s| s.extract::<usize>())
        .unwrap_or(if kind == "audio" { 2 } else { 3 });
    let elements = if kind == "audio" {
        let length = *shape
            .last()
            .ok_or_else(|| PyValueError::new_err("Invalid audio latent shape"))?;
        let length = if ratios.is_instance_of::<PyInt>() {
            length.saturating_mul(ratios.extract::<usize>()?)
        } else {
            length
        };
        shape
            .first()
            .copied()
            .unwrap_or(0)
            .saturating_mul(channels)
            .saturating_mul(length)
    } else {
        let axes = shape.get(2..).unwrap_or_default();
        let ratios: Vec<Py<PyAny>> = if ratios.is_instance_of::<pyo3::types::PyTuple>() {
            ratios.extract()?
        } else {
            axes.iter().map(|_| ratios.clone().unbind()).collect()
        };
        let mut pixels = 1usize;
        for (ratio, axis) in ratios.iter().zip(axes) {
            let ratio = ratio.bind(vae.py());
            let size = if ratio.is_callable() {
                ratio.call1((*axis,))?.extract::<usize>()?
            } else {
                ratio.extract::<usize>()?.saturating_mul(*axis)
            };
            pixels = pixels.saturating_mul(size);
        }
        shape
            .first()
            .copied()
            .unwrap_or(0)
            .saturating_mul(channels)
            .saturating_mul(pixels)
    };
    vae_check_budget(elements, max_bytes)
}
#[pyfunction]
pub(super) fn vae_upload_relative(filename: &Bound<'_, PyAny>) -> PyResult<String> {
    let value = filename.extract::<String>().ok();
    let valid = value
        .as_deref()
        .and_then(|s| s.strip_prefix("remote-conditioning/"))
        .and_then(|s| s.strip_suffix("/samples.pt [temp]"))
        .is_some_and(|id| {
            (1..=64).contains(&id.len())
                && id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        });
    if !valid {
        return Err(PyValueError::new_err(
            "Remote VAE latent must belong to a request-scoped temp upload",
        ));
    }
    Ok(value.unwrap().trim_end_matches(" [temp]").into())
}

#[pyfunction]
fn vae_decode_validation(
    py: Python<'_>,
    vae_name: &Bound<'_, PyAny>,
    output_mode: &str,
) -> PyResult<(bool, Py<PyAny>)> {
    let valid = vae_validate_model_choice(vae_name)?;
    if !model(vae_name) || output_mode == "tensor" {
        return Ok((false, valid));
    }
    if output_mode != "shared" {
        return Ok((
            false,
            "알 수 없는 결과 방식입니다."
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        ));
    }
    Ok((true, true.into_pyobject(py)?.to_owned().into_any().unbind()))
}
#[pyfunction]
fn vae_request_body(
    payload: &Bound<'_, PyAny>,
    fields: Vec<String>,
    message: &str,
) -> PyResult<Py<PyAny>> {
    let valid = payload.cast::<PyDict>().is_ok_and(|d| {
        d.len() == fields.len() && fields.iter().all(|k| d.contains(k).unwrap_or(false))
    });
    if !valid {
        return Err(PyValueError::new_err(message.to_owned()));
    }
    Ok(payload.clone().unbind())
}

#[pyfunction]
fn vae_exclusive_sources(
    prompt: &Bound<'_, PyAny>,
    unique_id: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let Ok(graph) = prompt.cast::<PyDict>() else {
        return Ok(true);
    };
    let key = unique_id.str()?;
    let mut links = Vec::new();
    if let Some(node) = graph.get_item(&key)? {
        if let Ok(inputs) = node
            .call_method1("get", ("inputs", PyDict::new(prompt.py())))?
            .cast::<PyDict>()
        {
            for name in ["video_path", "audio_path"] {
                if let Some(value) = inputs
                    .get_item(name)?
                    .filter(|v| v.is_instance_of::<pyo3::types::PyList>())
                {
                    links.push(value.unbind());
                }
            }
        }
    }
    for (id, node) in graph.iter() {
        if id.str()?.extract::<String>()? == key.extract::<String>()? {
            continue;
        }
        let inputs = node.call_method1("get", ("inputs", PyDict::new(prompt.py())))?;
        let inputs = inputs.cast::<PyDict>()?;
        for (_, value) in inputs.iter() {
            if !value.is_instance_of::<pyo3::types::PyList>() {
                continue;
            }
            for link in &links {
                if value.eq(link.bind(prompt.py()))? {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

pub(super) fn decode_defaults() -> Value {
    json!({"tiled":false,"tile_size":512,"overlap":64,"temporal_size":64,"temporal_overlap":8})
}
#[pyfunction]
fn vae_decode_defaults(py: Python<'_>) -> PyResult<Py<PyAny>> {
    to_python(py, &decode_defaults())
}
#[pyfunction]
fn vae_video_fields(py: Python<'_>) -> PyResult<Py<PyAny>> {
    to_python(
        py,
        &json!({
         "fps":["FLOAT",{"default":24.0,"min":1.0,"max":120.0}],
         "audio_vae_name":[[""],{"default":"","tooltip":"오디오 VAE · 선택 사항"}],
         "crf":["INT",{"default":18,"min":0,"max":51,"advanced":true}],
         "tiled":["BOOLEAN",{"default":false,"advanced":true}]
        }),
    )
}

//! Request transaction orchestration. HTTP/Torch/Comfy calls are external effects.
use super::{codec, policy, secure_fs, store};
use crate::{from_python, to_python};
use pyo3::{
    exceptions::{PyException, PyOSError, PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict},
};
use serde_json::{Value, json};
use std::fs;
fn uuid(py: Python<'_>) -> PyResult<String> {
    py.import("uuid")?
        .call_method0("uuid4")?
        .getattr("hex")?
        .extract()
}
fn warn(py: Python<'_>, message: &str, err: &PyErr) {
    let _ = py
        .import("logging")
        .and_then(|m| m.call_method1("getLogger", ("RemoteWorker",)))
        .and_then(|logger| logger.call_method1("warning", (format!("{message}: {err}"),)));
}
fn discard(api: &Bound<'_, PyAny>, method: &str, id: &str) {
    if let Err(err) = api.call_method1(method, (id,)) {
        warn(api.py(), "Could not remove VAE transfer input/output", &err);
    }
}
fn cancel(api: &Bound<'_, PyAny>, id: &str) {
    if let Err(err) = api.call_method1("cancel_prompt", (id,)) {
        warn(api.py(), "Could not cancel VAE peer request", &err);
    }
}
fn contextual(py: Python<'_>, error: PyErr, message: String) -> PyErr {
    let cancellation = !error.is_instance_of::<PyException>(py)
        || error
            .get_type(py)
            .getattr("__mro__")
            .and_then(|mro| {
                mro.try_iter()?
                    .map(|kind| kind?.getattr("__name__")?.extract::<String>())
                    .collect::<PyResult<Vec<_>>>()
            })
            .is_ok_and(|names| {
                names.iter().any(|name| {
                    matches!(
                        name.as_str(),
                        "RequestCancelled" | "InterruptProcessingException" | "CancelledError"
                    )
                })
            });
    if cancellation {
        let _ = error.value(py).call_method1("add_note", (&message,));
        return error;
    }
    let contextual = PyRuntimeError::new_err(format!("{message}: {error}"));
    contextual.set_cause(py, Some(error));
    contextual
}
fn prompt(api: &Bound<'_, PyAny>, graph: Py<PyAny>) -> PyResult<String> {
    let id = api
        .call_method1("submit", (graph,))?
        .extract::<String>()
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| PyRuntimeError::new_err("RemoteWorker API returned no prompt_id"))?;
    Ok(id)
}
fn with_prompt<T, F>(api: &Bound<'_, PyAny>, graph: Py<PyAny>, use_history: F) -> PyResult<T>
where
    F: FnOnce(&str, &Bound<'_, PyAny>) -> PyResult<T>,
{
    let id = prompt(api, graph)?;
    let result = api
        .call_method1("wait_for_history", (&id,))
        .and_then(|history| use_history(&id, &history));
    if result.is_err() {
        cancel(api, &id);
    }
    result
}
#[pyfunction]
#[pyo3(signature=(api,*,samples,vae_name,kind,options,memory_policy,max_bytes))]
fn vae_execute_decode(
    api: &Bound<'_, PyAny>,
    samples: &Bound<'_, PyAny>,
    vae_name: &Bound<'_, PyAny>,
    kind: &str,
    options: &Bound<'_, PyAny>,
    memory_policy: &str,
    max_bytes: usize,
) -> PyResult<Py<PyAny>> {
    let py = api.py();
    policy::vae_validate_kind(kind, false)?;
    policy::DecodeSettings::parse(options.cast::<PyDict>()?)?;
    policy::vae_preflight(api, vae_name, "RemoteVAEDecodeWorker", memory_policy)?;
    api.call_method1("require_node", ("SaveRemoteVAEOutput",))?;
    codec::latent(samples)?;
    let data = codec::vae_dumps(samples, max_bytes)?;
    let id = uuid(py)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("request_id", &id)?;
    kwargs.set_item("filename", "samples.pt")?;
    kwargs.set_item("data", data)?;
    kwargs.set_item("content_type", "application/octet-stream")?;
    let filename = api.call_method("upload_file", (), Some(&kwargs))?;
    let result = (|| {
        let graph = policy::vae_build_workflow(
            py,
            vae_name.extract()?,
            filename.extract()?,
            kind,
            options,
            &id,
            memory_policy,
        )?;
        let result = with_prompt(api, graph, |prompt_id, history| {
            let paths = crate::artifacts::extract_artifact_paths(
                history,
                prompt_id,
                &id,
                vec![(
                    "result".into(),
                    "save".into(),
                    "vae_output".into(),
                    ".pt".into(),
                )],
            )?;
            let path = paths.bind(py).get_item("result")?;
            let blob = api
                .call_method1("download_artifact", (path,))?
                .cast_into::<PyBytes>()?;
            let value = codec::vae_loads(&blob, max_bytes)?;
            codec::vae_validate_result(value.bind(py), kind)
        });
        discard(api, "discard_vae_outputs", &id);
        result
    })();
    discard(api, "discard_inputs", &id);
    result
}
#[pyfunction]
#[pyo3(signature=(api,*,vae_name,kind,server_url))]
fn vae_schedule(
    api: &Bound<'_, PyAny>,
    vae_name: &str,
    kind: &str,
    server_url: &str,
) -> PyResult<String> {
    let id = uuid(api.py())?;
    let graph = policy::vae_warmup_graph(api.py(), vae_name, kind, &id)?;
    let prompt = api.call_method1("submit", (graph,))?;
    policy::vae_ticket(api.py(), &prompt, &id, server_url, vae_name, kind)
}
#[pyfunction]
#[pyo3(signature=(api,ticket,*,vae_name,kind,server_url))]
fn vae_finish(
    api: &Bound<'_, PyAny>,
    ticket: &Bound<'_, PyAny>,
    vae_name: &str,
    kind: &str,
    server_url: &str,
) -> PyResult<()> {
    let py = api.py();
    let Some(data) = policy::vae_validate_ticket(py, ticket, vae_name, kind, server_url)? else {
        return Ok(());
    };
    let data = data.bind(py);
    let prompt_id: String = data.get_item("prompt_id")?.extract()?;
    let request_id: String = data.get_item("request_id")?.extract()?;
    let result = (|| {
        let history = api.call_method1("wait_for_history", (&prompt_id,))?;
        let paths = crate::artifacts::extract_artifact_paths(
            &history,
            &prompt_id,
            &request_id,
            vec![("prompt".into(), "save".into(), "text".into(), ".txt".into())],
        )?;
        let result = api
            .call_method1("download_artifact", (paths.bind(py).get_item("prompt")?,))
            .and_then(|blob| {
                if blob.extract::<Vec<u8>>()? != b"ready" {
                    Err(PyRuntimeError::new_err("Remote VAE warmup did not finish"))
                } else {
                    Ok(())
                }
            });
        if api.hasattr("discard_vae_outputs")? {
            discard(api, "discard_vae_outputs", &request_id);
        }
        result
    })();
    if result.is_err() {
        cancel(api, &prompt_id);
    }
    result
}
#[pyfunction]
#[pyo3(signature=(api,*,vae_name,kind,server_url,background=false))]
fn vae_execute_warmup(
    api: &Bound<'_, PyAny>,
    vae_name: &Bound<'_, PyAny>,
    kind: &str,
    server_url: &str,
    background: bool,
) -> PyResult<String> {
    policy::vae_validate_kind(kind, true)?;
    policy::vae_preflight(api, vae_name, "RemoteVAEWarmupWorker", "keep")?;
    api.call_method1("require_node", ("SaveRemoteText",))?;
    let ticket = vae_schedule(api, vae_name.extract()?, kind, server_url)?;
    if !background {
        let bound = ticket.clone().into_pyobject(api.py())?.into_any();
        vae_finish(api, &bound, vae_name.extract()?, kind, server_url)?;
    }
    Ok(ticket)
}
#[pyfunction]
fn vae_receive_probe(
    py: Python<'_>,
    request_id: &str,
    token: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let token = token
        .extract::<String>()
        .ok()
        .filter(|s| s.chars().count() == 48)
        .ok_or_else(|| PyValueError::new_err("Invalid shared storage challenge"))?;
    let path = store::path("temp", request_id, "probe.json")?;
    let value: Value = serde_json::from_slice(&fs::read(&path)?)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if value.get("token").and_then(Value::as_str) != Some(&token) {
        return Err(PyValueError::new_err(
            "처리 서버에서 호출 서버의 공유 파일을 읽을 수 없습니다.",
        ));
    }
    let result = json!({"token":token});
    store::write_json(py, &path.with_file_name("probe-ack.json"), &result)?;
    to_python(py, &result)
}
#[pyfunction]
fn vae_check_shared(api: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = api.py();
    let id = uuid(py)?;
    let token: String = py
        .import("secrets")?
        .call_method1("token_hex", (24,))?
        .extract()?;
    let probe = store::path("temp", &id, "probe.json")?;
    let ack = probe.with_file_name("probe-ack.json");
    let result = (|| {
        store::write_json(py, &probe, &json!({"token":token}))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            "payload",
            to_python(py, &json!({"request_id":id,"token":token}))?,
        )?;
        let result = from_python(&api.call_method(
            "_json",
            ("/remote-vae/shared/check",),
            Some(&kwargs),
        )?)?;
        let ack_value: Value = serde_json::from_slice(&fs::read(&ack)?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if result.get("token").and_then(Value::as_str) != Some(&token)
            || ack_value.get("token").and_then(Value::as_str) != Some(&token)
        {
            return Err(PyValueError::new_err(
                "공유 폴더 확인 응답이 일치하지 않습니다.",
            ));
        }
        let root = store::path("temp", &id, "probe.json")?
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        to_python(py, &json!({"root":root,"ready":true}))
    })();
    for p in [&probe, &ack] {
        store::remove_missing(p)?;
    }
    if let Some(parent) = probe.parent() {
        if parent.is_dir() && fs::read_dir(parent)?.next().is_none() {
            fs::remove_dir(parent)?;
        }
    }
    result.map_err(|err| {
        if err.is_instance_of::<PyOSError>(py)
            || err.is_instance_of::<PyValueError>(py)
            || err.is_instance_of::<PyRuntimeError>(py)
        {
            PyValueError::new_err(format!(
                "공유 폴더 연결 실패: 양쪽 서버의 공유 경로와 읽기·쓰기 권한을 확인하세요. {err}"
            ))
        } else {
            err
        }
    })
}
#[pyfunction]
fn vae_consume(
    py: Python<'_>,
    request_id: &str,
    reported: &Bound<'_, PyAny>,
    settings: &Bound<'_, PyAny>,
    check_cancelled: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let _lock = store::acquire(py, request_id)?;
    let options = from_python(settings)?;
    let current = store::read(request_id, Some(&options))?;
    let reported = from_python(reported)?;
    let Some(current) = current.filter(|c| c.document == reported) else {
        return Err(PyValueError::new_err(
            "공유 텐서 완료 정보가 일치하지 않습니다. 임시 파일을 보존합니다.",
        ));
    };
    check_cancelled.call0()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("map_location", "cpu")?;
    kwargs.set_item("weights_only", true)?;
    let value = py.import("torch")?.call_method(
        "load",
        (store::as_path(
            py,
            &secure_fs::descriptor_path(
                current
                    .media_file
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("Shared VAE tensor file is missing"))?,
            ),
        )?,),
        Some(&kwargs),
    )?;
    let value = codec::restore(&value)?;
    let kind = options
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| PyValueError::new_err("Unknown Remote VAE output kind"))?;
    let value = codec::vae_validate_result(value.bind(py), kind)?;
    check_cancelled.call0()?;
    store::remove_missing(&store::path("temp", request_id, "latent.pt")?)?;
    secure_fs::remove(&current.path, false)?;
    secure_fs::remove(&current.path.with_file_name("decoded-receipt.json"), false)?;
    Ok(value)
}
#[pyfunction]
fn vae_execute_shared(
    api: &Bound<'_, PyAny>,
    request_id: &str,
    options: &Bound<'_, PyAny>,
    server_url: &str,
    memory_policy: &str,
    warmup: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = api.py();
    let result = (|| {
        let values = options.cast::<PyDict>()?;
        let vae = values
            .get_item("vae_name")?
            .ok_or_else(|| PyValueError::new_err("Remote VAE requires a model name"))?;
        policy::vae_preflight(api, &vae, "SharedVAEVideoWorker", memory_policy)?;
        if let Some(audio) = values
            .get_item("audio_vae_name")?
            .filter(|s| s.is_truthy().unwrap_or(false))
        {
            policy::vae_preflight(api, &audio, "SharedVAEVideoWorker", "keep")?;
        }
        let kind = values
            .get_item("kind")?
            .and_then(|s| s.extract::<String>().ok())
            .unwrap_or("image".into());
        vae_finish(api, warmup, vae.extract()?, &kind, server_url)?;
        let graph = policy::vae_shared_workflow(py, request_id, options, memory_policy, false)?;
        with_prompt(api, graph, |id, history| {
            let outputs = crate::artifacts::successful_outputs(history, id)?;
            let reported = policy::vae_extract_receipt(outputs.bind(py), false)?;
            super::store::vae_acknowledge(py, request_id, reported.bind(py), options)
                .map(|path| path.into_pyobject(py).unwrap().into_any().unbind())
        })
    })();
    result.map_err(|err| {
        contextual(
            py,
            err,
            format!("공유 VAE 작업 실패 · 재시도 ID: {request_id} · 임시 latent 보존"),
        )
    })
}
#[pyfunction]
#[allow(clippy::too_many_arguments)] // Mirrors the stable ComfyUI/Python node ABI.
fn vae_execute_tensor(
    api: &Bound<'_, PyAny>,
    samples: &Bound<'_, PyAny>,
    vae_name: &Bound<'_, PyAny>,
    kind: &str,
    server_url: &str,
    memory_policy: &str,
    decode_options: &Bound<'_, PyDict>,
    request_id: &str,
    warmup: &Bound<'_, PyAny>,
    check_cancelled: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = api.py();
    let settings = policy::vae_tensor_options(py, vae_name, kind, decode_options)?;
    policy::vae_preflight(api, vae_name, "SharedVAETensorWorker", memory_policy)?;
    vae_check_shared(api)?;
    let request_id = if request_id.is_empty() {
        store::vae_save_latent(samples, None)?
    } else {
        request_id.into()
    };
    let result = (|| {
        vae_finish(api, warmup, vae_name.extract()?, kind, server_url)?;
        let graph =
            policy::vae_shared_workflow(py, &request_id, settings.bind(py), memory_policy, true)?;
        with_prompt(api, graph, |id, history| {
            let outputs = crate::artifacts::successful_outputs(history, id)?;
            let reported = policy::vae_extract_receipt(outputs.bind(py), true)?;
            vae_consume(
                py,
                &request_id,
                reported.bind(py),
                settings.bind(py),
                check_cancelled,
            )
        })
    })();
    result.map_err(|err| {
        contextual(
            py,
            err,
            format!("공유 텐서 전달 실패 · 재시도 ID: {request_id} · 임시 파일 보존"),
        )
    })
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vae_decode, m)?)?;
    m.add_function(wrap_pyfunction!(vae_encode_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(vae_encode_video, m)?)?;
    m.add_function(wrap_pyfunction!(vae_run_decode, m)?)?;
    m.add_function(wrap_pyfunction!(vae_load_upload, m)?)?;
    m.add_function(wrap_pyfunction!(vae_discard_output, m)?)?;
    m.add_function(wrap_pyfunction!(vae_execute_decode, m)?)?;
    m.add_function(wrap_pyfunction!(vae_schedule, m)?)?;
    m.add_function(wrap_pyfunction!(vae_finish, m)?)?;
    m.add_function(wrap_pyfunction!(vae_execute_warmup, m)?)?;
    m.add_function(wrap_pyfunction!(vae_receive_probe, m)?)?;
    m.add_function(wrap_pyfunction!(vae_check_shared, m)?)?;
    m.add_function(wrap_pyfunction!(vae_consume, m)?)?;
    m.add_function(wrap_pyfunction!(vae_execute_shared, m)?)?;
    m.add_function(wrap_pyfunction!(vae_execute_tensor, m)?)?;
    Ok(())
}

#[pyfunction]
fn vae_load_upload(
    filename: &Bound<'_, PyAny>,
    temp_root: &Bound<'_, PyAny>,
    max_bytes: usize,
) -> PyResult<Py<PyAny>> {
    let relative = policy::vae_upload_relative(filename)?;
    let root = fs::canonicalize(std::path::PathBuf::from(temp_root.str()?.to_str()?))?;
    let path = fs::canonicalize(root.join(relative))?;
    if !path.starts_with(&root) {
        return Err(PyValueError::new_err(
            "Remote VAE latent path escapes temp storage",
        ));
    }
    use std::io::Read;
    let mut data = Vec::new();
    fs::File::open(path)?
        .take(max_bytes.saturating_add(1) as u64)
        .read_to_end(&mut data)?;
    let py = filename.py();
    let data = PyBytes::new(py, &data);
    let value = codec::vae_loads(&data, max_bytes)?;
    codec::latent(value.bind(py))?;
    Ok(value)
}
#[pyfunction]
fn vae_discard_output(
    request_id: &Bound<'_, PyAny>,
    output_root: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let id = request_id
        .extract::<String>()
        .ok()
        .filter(|s| crate::protocol::RequestId::parse(s).is_some())
        .ok_or_else(|| PyValueError::new_err("Invalid Remote VAE request ID"))?;
    let py = request_id.py();
    let root = py
        .import("pathlib")?
        .getattr("Path")?
        .call1((output_root,))?
        .call_method0("resolve")?;
    let root = std::path::PathBuf::from(root.str()?.to_str()?);
    let path = root.join("remote-conditioning").join(id);
    let resolved = py
        .import("pathlib")?
        .getattr("Path")?
        .call1((path.to_string_lossy().as_ref(),))?
        .call_method0("resolve")?;
    let resolved = std::path::PathBuf::from(resolved.str()?.to_str()?);
    if !resolved.starts_with(&root)
        || fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink())
    {
        return Err(PyValueError::new_err(
            "Remote VAE output directory escapes output storage",
        ));
    }
    let mut removed = false;
    if path.exists() {
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let digits = name
                .strip_prefix("decoded_")
                .and_then(|s| s.strip_suffix("_.pt"))
                .or_else(|| {
                    name.strip_prefix("warmup_")
                        .and_then(|s| s.strip_suffix("_.txt"))
                });
            let owned =
                digits.is_some_and(|d| d.len() >= 5 && d.bytes().all(|b| b.is_ascii_digit()));
            let metadata = fs::symlink_metadata(entry.path())?;
            if owned && metadata.is_file() && !metadata.file_type().is_symlink() {
                fs::remove_file(entry.path())?;
                removed = true;
            }
        }
        if fs::read_dir(&path)?.next().is_none() {
            fs::remove_dir(&path)?;
        }
    }
    Ok(removed)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Mirrors the stable ComfyUI/Python node ABI.
fn vae_run_decode(
    api: &Bound<'_, PyAny>,
    samples: &Bound<'_, PyAny>,
    vae_name: &Bound<'_, PyAny>,
    kind: &str,
    server_url: &str,
    memory_policy: &str,
    options: &Bound<'_, PyDict>,
    warmup: &Bound<'_, PyAny>,
    output_mode: &str,
    request_id: &str,
    max_bytes: usize,
    check_cancelled: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    match output_mode {
        "shared" => vae_execute_tensor(
            api,
            samples,
            vae_name,
            kind,
            server_url,
            memory_policy,
            options,
            request_id,
            warmup,
            check_cancelled,
        ),
        "tensor" => {
            vae_finish(api, warmup, vae_name.extract()?, kind, server_url)?;
            vae_execute_decode(
                api,
                samples,
                vae_name,
                kind,
                options.as_any(),
                memory_policy,
                max_bytes,
            )
        }
        _ => Err(PyValueError::new_err("알 수 없는 결과 방식입니다.")),
    }
}

#[pyfunction]
#[pyo3(signature=(vae,samples,kind,tiled,tile_size,overlap,temporal_size,temporal_overlap,*,transfer=true,max_bytes=536870912))]
#[allow(clippy::too_many_arguments)] // Mirrors the stable ComfyUI/Python node ABI.
fn vae_decode(
    vae: &Bound<'_, PyAny>,
    samples: &Bound<'_, PyAny>,
    kind: &str,
    tiled: &Bound<'_, PyAny>,
    tile_size: &Bound<'_, PyAny>,
    overlap: &Bound<'_, PyAny>,
    temporal_size: &Bound<'_, PyAny>,
    temporal_overlap: &Bound<'_, PyAny>,
    transfer: bool,
    max_bytes: usize,
) -> PyResult<Py<PyAny>> {
    let fields = PyDict::new(vae.py());
    for (key, value) in [
        ("tiled", tiled),
        ("tile_size", tile_size),
        ("overlap", overlap),
        ("temporal_size", temporal_size),
        ("temporal_overlap", temporal_overlap),
    ] {
        fields.set_item(key, value)?;
    }
    let options = policy::DecodeSettings::parse(&fields)?;
    policy::vae_validate_kind(kind, false)?;
    policy::vae_check_channels(
        kind,
        vae.getattr("output_channels")
            .ok()
            .and_then(|v| v.extract().ok()),
    )?;
    if transfer {
        policy::vae_output_budget(vae, samples, kind, max_bytes)?;
    }
    let py = vae.py();
    if kind == "audio" {
        return Ok(py
            .import("comfy_extras.nodes_audio")?
            .call_method1("vae_decode_audio", (vae, samples))?
            .unbind());
    }
    let nodes = py.import("nodes")?;
    let result = if options.tiled {
        nodes.getattr("VAEDecodeTiled")?.call0()?.call_method1(
            "decode",
            (
                vae,
                samples,
                options.tile_size,
                options.overlap,
                options.temporal_size,
                options.temporal_overlap,
            ),
        )?
    } else {
        nodes
            .getattr("VAEDecode")?
            .call0()?
            .call_method1("decode", (vae, samples))?
    };
    Ok(result.get_item(0)?.unbind())
}
fn decode_options<'py>(
    py: Python<'py>,
    settings: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let defaults = to_python(py, &policy::decode_defaults())?;
    let options = defaults.bind(py).cast::<PyDict>()?.copy()?;
    if let Some(settings) = settings {
        for (key, _) in defaults.bind(py).cast::<PyDict>()?.iter() {
            if let Ok(value) = settings.get_item(key.clone()) {
                options.set_item(key, value)?;
            }
        }
    }
    let checked = policy::DecodeSettings::parse(&options)?;
    let checked = to_python(py, &checked.value())?;
    let options = checked.bind(py).cast::<PyDict>()?.copy()?;
    options.set_item("transfer", false)?;
    Ok(options)
}
#[pyfunction]
fn vae_encode_tensor(
    samples: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    settings: &Bound<'_, PyAny>,
    load_vae: &Bound<'_, PyAny>,
    decode: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let vae = load_vae.call1((settings.get_item("vae_name")?,))?;
    let kind: String = settings.get_item("kind")?.extract()?;
    let result = decode.call(
        (vae, samples, &kind),
        Some(&decode_options(samples.py(), Some(settings))?),
    )?;
    let result = codec::vae_validate_result(&result, &kind)?;
    let result = codec::cpu(result.bind(samples.py()))?;
    samples
        .py()
        .import("torch")?
        .call_method1("save", (result, path))?;
    to_python(samples.py(), &json!({"kind":kind}))
}
#[pyfunction]
fn vae_encode_video(
    samples: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
    load_vae: &Bound<'_, PyAny>,
    decode: &Bound<'_, PyAny>,
    encode_wav: &Bound<'_, PyAny>,
    encode_mp4: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = samples.py();
    let vae = load_vae.call1((options.get_item("vae_name")?,))?;
    let audio = options
        .call_method1("get", ("kind",))?
        .extract::<String>()
        .is_ok_and(|v| v == "audio");
    if audio {
        let result = decode.call((vae, samples, "audio"), Some(&decode_options(py, None)?))?;
        let waveform = result
            .get_item("waveform")?
            .call_method0("detach")?
            .call_method0("cpu")?;
        let shape: Vec<usize> = waveform.getattr("shape")?.extract()?;
        policy::vae_check_audio_shape(shape[0], shape[1])?;
        encode_wav.call1((&result, waveform, path))?;
        let sample_rate: usize = result.get_item("sample_rate")?.extract()?;
        return to_python(
            py,
            &json!({"audio":true,"sample_rate":sample_rate,"samples":shape.last()}),
        );
    }
    let settings = decode_options(py, None)?;
    settings.set_item(
        "tiled",
        policy::strict_bool(&options.get_item("tiled")?, "tiled")?,
    )?;
    let frames = decode.call((vae, samples, "image"), Some(&settings))?;
    let mut audio = samples.call_method1("get", ("remote_vae_audio",))?;
    let audio_vae = options.get_item("audio_vae_name")?;
    if audio_vae.is_truthy()? {
        let vae = load_vae.call1((audio_vae,))?;
        audio = decode.call((vae, samples, "audio"), Some(&decode_options(py, None)?))?;
    }
    encode_mp4.call1((&frames, &audio, path, options))?;
    let shape: Vec<usize> = frames.getattr("shape")?.extract()?;
    to_python(
        py,
        &json!({"frames":shape[0],"width":shape[2],"height":shape[1],"audio":!audio.is_none()}),
    )
}

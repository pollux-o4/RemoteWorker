//! HTTP/UI trust boundaries. Adapters perform IO; Rust validates its contracts.
use crate::protocol::RequestId;
use crate::{from_python, to_python};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBool, PyBytes, PyDict, PyInt, PyList, PyTuple},
};
use serde_json::{Value, json};
fn invalid(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}
fn exact(value: &Value, fields: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|m| m.len() == fields.len() && fields.iter().all(|s| m.contains_key(*s)))
}
#[pyfunction]
#[pyo3(signature=(payload,profile=false))]
fn proxy_config(payload: &Bound<'_, PyAny>, profile: bool) -> PyResult<(String, i64)> {
    let value = from_python(payload)?;
    if !value.is_object() && !profile {
        return Err(invalid("request body must be a JSON object"));
    }
    if !exact(&value, &["server_url", "timeout_seconds"]) {
        return Err(invalid(
            "request body must contain only server_url and timeout_seconds",
        ));
    }
    let url = value["server_url"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid("server_url must be a non-empty string"))?;
    let timeout = value["timeout_seconds"].as_i64().ok_or_else(|| {
        invalid(if profile {
            "timeout_seconds must be between 5 and 30"
        } else {
            "timeout_seconds must be an integer"
        })
    })?;
    if !(5..=30).contains(&timeout) {
        return Err(invalid("timeout_seconds must be between 5 and 30"));
    }
    Ok((url.into(), timeout))
}
#[pyfunction]
fn proxy_options(options: &Bound<'_, PyAny>, fields: Vec<String>) -> PyResult<Py<PyAny>> {
    let value = from_python(options)?;
    if !value
        .as_object()
        .is_some_and(|m| m.len() == fields.len() && fields.iter().all(|s| m.contains_key(s)))
    {
        return Err(invalid(format!(
            "upstream options must contain exactly {}",
            if fields.len() == 1 {
                fields[0].clone()
            } else {
                "clip_name, clip_type, and device".into()
            }
        )));
    }
    for field in &fields {
        if !value[field].as_array().is_some_and(|a| {
            !a.is_empty() && a.iter().all(|v| v.as_str().is_some_and(|s| !s.is_empty()))
        }) {
            return Err(invalid(format!(
                "upstream {field} options must be a non-empty string array"
            )));
        }
    }
    to_python(options.py(), &value)
}
#[pyfunction]
fn validate_request_body(payload: &Bound<'_, PyAny>) -> PyResult<String> {
    let value = from_python(payload)?;
    if !exact(&value, &["request_id"]) {
        return Err(invalid("request body must contain only request_id"));
    }
    let id = value["request_id"]
        .as_str()
        .filter(|s| RequestId::parse(s).is_some())
        .ok_or_else(|| invalid("request_id must be 1-64 URL-safe characters"))?;
    Ok(id.into())
}
#[pyfunction]
#[pyo3(signature=(payload,proxy=false))]
fn profile_action(payload: &Bound<'_, PyAny>, proxy: bool) -> PyResult<String> {
    let value = from_python(payload)?;
    if !value.is_object() {
        return Err(invalid("request body must be a JSON object"));
    }
    let action = value["action"].as_str();
    let extra = if proxy {
        vec!["server_url", "timeout_seconds"]
    } else {
        vec![]
    };
    for (action_name, field) in [("upsert", "profile"), ("delete", "profile_id")] {
        let mut fields = extra.clone();
        fields.extend(["action", field]);
        if action == Some(action_name) && exact(&value, &fields) {
            return Ok(action_name.into());
        }
    }
    Err(invalid(
        "request action must be upsert with profile or delete with profile_id",
    ))
}
#[pyfunction]
fn validate_available_profile(
    profile: &Bound<'_, PyAny>,
    names: Vec<String>,
    types: Vec<String>,
    devices: Vec<String>,
) -> PyResult<()> {
    for (field, choices, reason) in [
        ("clip_name", names, "is not available"),
        ("clip_type", types, "is not supported"),
        ("device", devices, "is not supported"),
    ] {
        let value: String = profile.getattr(field)?.extract()?;
        if !choices.contains(&value) {
            return Err(invalid(format!(
                "Prompt enhancer profile {field} {reason} on this peer runtime: {value}"
            )));
        }
    }
    Ok(())
}
#[pyfunction]
fn boolean_response(response: &Bound<'_, PyAny>, field: &str, message: &str) -> PyResult<bool> {
    let Some(dict) = response.cast::<PyDict>().ok() else {
        return Err(PyRuntimeError::new_err(message.to_owned()));
    };
    if dict.len() != 1 {
        return Err(PyRuntimeError::new_err(message.to_owned()));
    }
    let value = dict
        .get_item(field)?
        .filter(|v| v.is_instance_of::<PyBool>())
        .ok_or_else(|| PyRuntimeError::new_err(message.to_owned()))?;
    value.extract()
}
#[pyfunction]
fn object_response(response: &Bound<'_, PyAny>, path: &str) -> PyResult<()> {
    if !response.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err(format!(
            "Remote Conditioning returned a non-object JSON response from {path}"
        )));
    }
    Ok(())
}
#[pyfunction]
fn prompt_response(response: &Bound<'_, PyAny>) -> PyResult<String> {
    let value = from_python(response)?;
    value["prompt_id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Remote Conditioning prompt was rejected: {}",
                value.get("error").unwrap_or(&value)
            ))
        })
}
#[pyfunction]
fn completed_history(history: &Bound<'_, PyAny>, prompt_id: &str) -> PyResult<bool> {
    let value = from_python(history)?;
    Ok(value
        .get(prompt_id)
        .and_then(|e| e.pointer("/status/status_str"))
        .and_then(Value::as_str)
        .is_some_and(|s| matches!(s, "success" | "error")))
}
#[pyfunction]
fn validate_upload<'py>(
    py: Python<'py>,
    request_id: &Bound<'py, PyAny>,
    filename: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    content_type: &Bound<'py, PyAny>,
    max_bytes: usize,
) -> PyResult<(String, String, String)> {
    let id = request_id
        .extract::<String>()
        .ok()
        .filter(|s| RequestId::parse(s).is_some())
        .ok_or_else(|| {
            invalid("Remote Conditioning request_id must be 1-64 URL-safe characters")
        })?;
    let err = || invalid("Remote Conditioning media upload arguments are invalid");
    let filename = filename.extract::<String>().map_err(|_| err())?;
    if !(1..=128).contains(&filename.len())
        || !filename.as_bytes()[0].is_ascii_alphanumeric()
        || !filename
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
    {
        return Err(err());
    }
    let mime = content_type.extract::<String>().map_err(|_| err())?;
    let allowed = |s: &str| {
        !s.is_empty()
            && s.len() <= 128
            && s.as_bytes()[0].is_ascii_alphanumeric()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$&^_.+/-".contains(&b))
    };
    if !mime
        .split_once('/')
        .is_some_and(|(a, b)| allowed(a) && allowed(b))
    {
        return Err(err());
    }
    let bytes = data.cast::<PyBytes>().map_err(|_| err())?;
    if bytes.as_bytes().is_empty() || bytes.as_bytes().len() > max_bytes {
        return Err(err());
    }
    let _ = py;
    Ok((id, filename, mime))
}
#[pyfunction]
fn upload_response(
    response: &Bound<'_, PyAny>,
    request_id: &str,
    filename: &str,
) -> PyResult<String> {
    let value = from_python(response)?;
    let subfolder = format!("remote-conditioning/{request_id}");
    if value.get("name") != Some(&json!(filename))
        || value.get("subfolder") != Some(&json!(subfolder))
        || value.get("type") != Some(&json!("temp"))
    {
        return Err(PyRuntimeError::new_err(format!(
            "Remote Conditioning media upload returned an unexpected path: {value}"
        )));
    }
    Ok(format!("{subfolder}/{filename} [temp]"))
}
#[pyfunction]
fn image_socket_name(value: &Bound<'_, PyAny>) -> bool {
    value.extract::<String>().is_ok_and(|s| {
        s.strip_prefix("image_").is_some_and(|n| {
            !n.is_empty() && n.as_bytes()[0] != b'0' && n.bytes().all(|b| b.is_ascii_digit())
        })
    })
}
#[pyfunction]
fn ordered_image_frames(
    image_count: &Bound<'_, PyAny>,
    images: &Bound<'_, PyDict>,
) -> PyResult<Py<PyList>> {
    let count = image_count
        .extract::<i64>()
        .ok()
        .filter(|v| (0..=64).contains(v) && !image_count.is_instance_of::<PyBool>())
        .ok_or_else(|| invalid("Remote Conditioning image_count must be 0 to 64"))?;
    let result = PyList::empty(images.py());
    for i in 1..=count {
        let name = format!("image_{i}");
        let batch = images
            .get_item(&name)?
            .filter(|v| !v.is_none())
            .ok_or_else(|| invalid(format!("Remote Conditioning requires connected {name}")))?;
        let shape = batch
            .getattr("shape")
            .ok()
            .filter(|s| {
                (s.is_instance_of::<PyList>() || s.is_instance_of::<PyTuple>())
                    && s.len().unwrap_or(0) == 4
            })
            .ok_or_else(|| {
                invalid(format!(
                    "Remote Conditioning {name} must be a BHWC IMAGE tensor"
                ))
            })?;
        let size = shape.extract::<Vec<i64>>().ok();
        if size
            .as_ref()
            .is_none_or(|s| s[..3].iter().any(|v| *v <= 0) || !matches!(s[3], 3 | 4))
            || shape
                .try_iter()?
                .take(3)
                .any(|v| v.is_ok_and(|v| !v.is_instance_of::<PyInt>()))
        {
            return Err(invalid(format!(
                "Remote Conditioning {name} has an invalid IMAGE shape: {}",
                shape.repr()?
            )));
        }
        let size = size.unwrap();
        if result.len() + size[0] as usize > 64 {
            return Err(invalid(
                "Remote Conditioning accepts at most 64 image attachments per request",
            ));
        }
        for index in 0..size[0] {
            result.append(batch.get_item(index)?)?;
        }
    }
    Ok(result.unbind())
}
#[pyfunction]
fn validate_text(text: &Bound<'_, PyAny>, message: &str) -> PyResult<()> {
    if !text
        .extract::<String>()
        .is_ok_and(|s| !s.trim().is_empty() && !s.contains('\0'))
    {
        return Err(invalid(message));
    }
    Ok(())
}
#[pyfunction]
fn validate_latent_dictionary(samples: &Bound<'_, PyAny>, message: &str) -> PyResult<()> {
    if !samples
        .cast::<PyDict>()
        .is_ok_and(|d| d.contains("samples").unwrap_or(false))
    {
        return Err(invalid(message));
    }
    Ok(())
}
#[pyfunction]
fn positive_env(raw: Option<String>, default: u64) -> u64 {
    raw.and_then(|s| s.trim().replace('_', "").parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(discard_request_inputs_native, m)?)?;
    m.add_function(wrap_pyfunction!(check_artifact_budget, m)?)?;
    m.add_function(wrap_pyfunction!(require_node_response, m)?)?;
    m.add_function(wrap_pyfunction!(upload_image_index, m)?)?;
    m.add_function(wrap_pyfunction!(upload_multipart, m)?)?;
    m.add_function(wrap_pyfunction!(download_query, m)?)?;
    m.add_function(wrap_pyfunction!(validate_prompt_id, m)?)?;
    m.add_function(wrap_pyfunction!(image_array_mode, m)?)?;
    m.add_function(wrap_pyfunction!(check_image_budget, m)?)?;
    m.add_function(wrap_pyfunction!(check_cancelled_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_clip_selection_native, m)?)?;
    m.add_function(wrap_pyfunction!(proxy_config, m)?)?;
    m.add_function(wrap_pyfunction!(proxy_options, m)?)?;
    m.add_function(wrap_pyfunction!(validate_request_body, m)?)?;
    m.add_function(wrap_pyfunction!(profile_action, m)?)?;
    m.add_function(wrap_pyfunction!(validate_available_profile, m)?)?;
    m.add_function(wrap_pyfunction!(boolean_response, m)?)?;
    m.add_function(wrap_pyfunction!(object_response, m)?)?;
    m.add_function(wrap_pyfunction!(prompt_response, m)?)?;
    m.add_function(wrap_pyfunction!(completed_history, m)?)?;
    m.add_function(wrap_pyfunction!(validate_upload, m)?)?;
    m.add_function(wrap_pyfunction!(upload_response, m)?)?;
    m.add_function(wrap_pyfunction!(image_socket_name, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_image_frames, m)?)?;
    m.add_function(wrap_pyfunction!(validate_text, m)?)?;
    m.add_function(wrap_pyfunction!(validate_latent_dictionary, m)?)?;
    m.add_function(wrap_pyfunction!(positive_env, m)?)?;
    Ok(())
}

#[pyfunction]
fn discard_request_inputs_native(
    request_id: &Bound<'_, PyAny>,
    temp_directory: std::path::PathBuf,
) -> PyResult<bool> {
    use std::{fs, path::Path};
    let id = request_id
        .extract::<String>()
        .ok()
        .filter(|s| RequestId::parse(s).is_some())
        .ok_or_else(|| invalid("request_id must be 1-64 URL-safe characters"))?;
    let root = fs::canonicalize(&temp_directory)?;
    let base = root.join("remote-conditioning");
    let target = base.join(id);
    // Never follow a replaced request/root symlink during scoped cleanup.
    for path in [&base, &target] {
        match fs::symlink_metadata(path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(invalid(
                    "request directory is outside the Remote Conditioning temp root",
                ));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e.into()),
            _ => {}
        }
    }
    if !Path::new(&target).is_dir() {
        return Err(invalid("request temp target is not a directory"));
    }
    fs::remove_dir_all(target)?;
    Ok(true)
}

#[pyfunction]
fn check_artifact_budget(length: usize, limit: usize) -> PyResult<()> {
    if length > limit {
        let actual = (length as f64 / 1048576.0 * 100.0).round() / 100.0;
        return Err(PyRuntimeError::new_err(format!(
            "Remote Conditioning artifact exceeds the {} MiB limit ({actual:?} MiB received)",
            limit / 1048576
        )));
    }
    Ok(())
}
#[pyfunction]
fn require_node_response<'py>(
    info: &Bound<'py, PyAny>,
    node_name: &str,
) -> PyResult<Bound<'py, PyDict>> {
    info.get_item(node_name)
        .and_then(|v| Ok(v.cast_into::<PyDict>()?))
        .map_err(|_| {
            PyRuntimeError::new_err(format!(
                "Remote Conditioning peer does not provide {node_name}"
            ))
        })
}
#[pyfunction]
fn upload_image_index(index: &Bound<'_, PyAny>) -> PyResult<u64> {
    index
        .extract::<u64>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("Remote Conditioning image upload arguments are invalid"))
}
#[pyfunction]
fn upload_multipart<'py>(
    py: Python<'py>,
    request_id: &Bound<'py, PyAny>,
    filename: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    content_type: &Bound<'py, PyAny>,
    max_bytes: usize,
    boundary: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let (id, filename, mime) =
        validate_upload(py, request_id, filename, data, content_type, max_bytes)?;
    let mut body = Vec::<u8>::new();
    for (name, value) in [
        ("type", "temp".into()),
        ("subfolder", format!("remote-conditioning/{id}")),
        ("overwrite", "true".into()),
    ] {
        body.extend(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n").as_bytes());
    body.extend(data.cast::<PyBytes>()?.as_bytes());
    body.extend(format!("\r\n--{boundary}--\r\n").as_bytes());
    Ok(PyBytes::new(py, &body))
}
#[pyfunction]
fn download_query(artifact: &str) -> PyResult<String> {
    let Some((folder, file)) = artifact.rsplit_once('/') else {
        return Err(invalid("not enough values to unpack (expected 2, got 1)"));
    };
    Ok(format!(
        "/view?{}",
        url::form_urlencoded::Serializer::new(String::new())
            .append_pair("filename", file)
            .append_pair("subfolder", folder)
            .append_pair("type", "output")
            .finish()
    ))
}
#[pyfunction]
fn validate_prompt_id(prompt_id: &Bound<'_, PyAny>) -> PyResult<String> {
    prompt_id
        .extract::<String>()
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("Remote Conditioning prompt_id must be a non-empty string"))
}

#[pyfunction]
fn image_array_mode(shape: &Bound<'_, PyAny>, finite: bool) -> PyResult<&'static str> {
    let dimensions = shape.extract::<Vec<usize>>().ok();
    let valid = dimensions
        .as_ref()
        .is_some_and(|s| s.len() == 3 && matches!(s[2], 3 | 4))
        && finite;
    if !valid {
        return Err(invalid(format!(
            "Remote Conditioning image attachment has invalid HWC data: {}",
            shape.repr()?
        )));
    }
    Ok(if dimensions.unwrap()[2] == 4 {
        "RGBA"
    } else {
        "RGB"
    })
}
#[pyfunction]
fn check_image_budget(length: usize, limit: usize) -> PyResult<()> {
    if length == 0 || length > limit {
        return Err(invalid(
            "Remote Conditioning encoded image exceeds the 32 MiB limit",
        ));
    }
    Ok(())
}

#[pyfunction]
fn check_cancelled_native(check: &Bound<'_, PyAny>, exception: &Bound<'_, PyAny>) -> PyResult<()> {
    if !check.is_none() && check.call0()?.is_truthy()? {
        return Err(PyErr::from_value(
            exception.call1(("Queued preparation request was cancelled or changed",))?,
        ));
    }
    Ok(())
}
#[pyfunction]
fn validate_clip_selection_native(
    clip: &Bound<'_, PyAny>,
    selection_type: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if !clip.is_instance(selection_type)? {
        return Err(invalid(
            "Remote Conditioning CLIP selection requires non-empty clip_name, clip_type, and device",
        ));
    }
    crate::protocol::validate_clip_values(clip)
}

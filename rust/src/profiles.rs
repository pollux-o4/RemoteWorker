//! Prompt profile and deployment defaults schema policy.
use crate::{from_python, to_python};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::{Value, json};
fn invalid(s: impl Into<String>) -> PyErr {
    PyValueError::new_err(s.into())
}
#[pyfunction]
fn validate_defaults_native(payload: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let v = from_python(payload)?;
    let fields = [
        "server_url",
        "conditioning_clip",
        "conditioning_clip_type",
        "qwen_clip",
        "qwen_vae",
        "minimax_clip",
        "minimax_video_vae",
        "minimax_audio_vae",
        "prompt_profile",
    ];
    let map = v
        .as_object()
        .filter(|m| m.keys().all(|k| fields.contains(&k.as_str())))
        .ok_or_else(|| invalid("RemoteWorker local settings contain unsupported fields"))?;
    if map
        .iter()
        .any(|(k, v)| k != "prompt_profile" && !v.is_string())
    {
        return Err(invalid("RemoteWorker default overrides must be strings"));
    }
    if map.get("prompt_profile").is_some_and(|v| !v.is_object()) {
        return Err(invalid(
            "RemoteWorker prompt_profile override must be an object",
        ));
    }
    to_python(payload.py(), &v)
}
#[pyfunction]
fn validate_profile_payload_native(
    payload: &Bound<'_, PyAny>,
    default_system_prompt: &str,
    default_media: Vec<String>,
) -> PyResult<Py<PyAny>> {
    if let Ok(media) = payload.get_item("accepted_media") {
        if !media.is_instance_of::<pyo3::types::PyList>() {
            return Err(invalid(
                "Prompt enhancer profile accepted_media must be a non-empty array",
            ));
        }
    }
    let mut v = from_python(payload)?;
    let legacy = [
        "id",
        "description",
        "clip_name",
        "clip_type",
        "device",
        "max_length",
    ];
    let current = [
        "id",
        "description",
        "clip_name",
        "clip_type",
        "device",
        "max_length",
        "system_prompt",
        "accepted_media",
    ];
    let map=v.as_object_mut().filter(|m|(m.len()==legacy.len()&&legacy.iter().all(|k|m.contains_key(*k)))||(m.len()==current.len()&&current.iter().all(|k|m.contains_key(*k)))).ok_or_else(||invalid("Prompt enhancer profile must contain id, description, clip_name, clip_type, device, max_length, system_prompt, and accepted_media"))?;
    map.entry("system_prompt")
        .or_insert(json!(default_system_prompt));
    map.entry("accepted_media").or_insert(json!(default_media));
    let id = map["id"].as_str().unwrap_or("");
    let bytes = id.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 64
        || !(bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        || !bytes
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-')
    {
        return Err(invalid(
            "Prompt enhancer profile id must be 1-64 lowercase letters, digits, or hyphens",
        ));
    }
    for key in [
        "description",
        "clip_name",
        "clip_type",
        "device",
        "system_prompt",
    ] {
        let text = map[key]
            .as_str()
            .filter(|s| !s.trim().is_empty() && !s.contains('\0'))
            .ok_or_else(|| {
                invalid(format!(
                    "Prompt enhancer profile {key} must be a non-empty string"
                ))
            })?
            .trim()
            .to_owned();
        if key == "description" && text.chars().count() > 480 {
            return Err(invalid(
                "Prompt enhancer profile description must be at most 480 characters",
            ));
        }
        map.insert(key.into(), text.into());
    }
    let media = map["accepted_media"]
        .as_array()
        .filter(|a| !a.is_empty() && a.iter().all(Value::is_string))
        .ok_or_else(|| {
            invalid("Prompt enhancer profile accepted_media must be a non-empty array")
        })?;
    let mut normalized = Vec::new();
    for item in media {
        let item = item.as_str().unwrap().trim();
        if !matches!(item, "image" | "video") {
            return Err(invalid(
                "Prompt enhancer profile accepted_media supports only image and video",
            ));
        }
        if !normalized.contains(&item) {
            normalized.push(item);
        }
    }
    if normalized.contains(&"video") && map["clip_type"] != "minimax" {
        return Err(invalid(
            "Prompt enhancer profile video support requires clip_type minimax",
        ));
    }
    let media = json!(normalized);
    map.insert("accepted_media".into(), media);
    if map["max_length"]
        .as_u64()
        .is_none_or(|n| !(16..=2048).contains(&n))
    {
        return Err(invalid(
            "Prompt enhancer profile max_length must be 16 to 2048",
        ));
    }
    to_python(payload.py(), &v)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_profile_payload_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_defaults_native, m)?)?;
    m.add_function(wrap_pyfunction!(profile_response_native, m)?)?;
    m.add_function(wrap_pyfunction!(profile_registry_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_profile_id_native, m)?)?;
    m.add_function(wrap_pyfunction!(profile_select_native, m)?)?;
    m.add_function(wrap_pyfunction!(profile_mutate_native, m)?)?;
    Ok(())
}

#[pyfunction]
fn profile_response_native(response: &Bound<'_, PyAny>, action: &str) -> PyResult<Py<PyAny>> {
    let v = from_python(response)?;
    let field = match action {
        "list" => "profiles",
        "upsert" => "profile",
        "delete" => "deleted",
        _ => return Err(invalid("Unknown profile response action")),
    };
    let valid = v.as_object().is_some_and(|m| {
        m.len() == 1
            && m.get(field).is_some_and(|v| match action {
                "list" => v.is_array(),
                "upsert" => v.is_object(),
                "delete" => v.is_boolean(),
                _ => false,
            })
    });
    if !valid {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Remote Prompt Enhancer profile {} is invalid",
            match action {
                "list" => "list",
                "upsert" => "upsert response",
                _ => "delete response",
            }
        )));
    }
    to_python(response.py(), &v[field])
}
#[pyfunction]
fn profile_registry_native(
    document: &Bound<'_, PyAny>,
    default_system_prompt: &str,
    default_media: Vec<String>,
) -> PyResult<Py<PyAny>> {
    let v = from_python(document)?;
    let invalid_registry = |s: &str| pyo3::exceptions::PyRuntimeError::new_err(s.to_owned());
    if !v
        .as_object()
        .is_some_and(|m| m.len() == 2 && m.contains_key("version") && m.contains_key("profiles"))
        || v["version"] != 1
    {
        return Err(invalid_registry(
            "Prompt enhancer profile registry has an unsupported schema",
        ));
    }
    let profiles = v["profiles"].as_array().ok_or_else(|| {
        invalid_registry("Prompt enhancer profile registry profiles must be an array")
    })?;
    let mut normalized = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for profile in profiles {
        let obj = to_python(document.py(), profile)?;
        let normalized_profile = validate_profile_payload_native(
            obj.bind(document.py()),
            default_system_prompt,
            default_media.clone(),
        )?;
        let profile = from_python(normalized_profile.bind(document.py()))?;
        if !ids.insert(profile["id"].as_str().unwrap().to_owned()) {
            return Err(invalid_registry(
                "Prompt enhancer profile registry contains duplicate ids",
            ));
        }
        normalized.push(profile);
    }
    normalized.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    to_python(document.py(), &json!(normalized))
}
#[pyfunction]
pub(crate) fn validate_profile_id_native(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let s = value.extract::<String>().unwrap_or_default();
    let bytes = s.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 64
        || !(bytes[0].is_ascii_lowercase() || bytes[0].is_ascii_digit())
        || !bytes
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-')
    {
        return Err(invalid("Prompt enhancer profile_id is invalid"));
    }
    Ok(s)
}
#[pyfunction]
fn profile_select_native(profiles: &Bound<'_, PyAny>, id: &str) -> PyResult<Py<PyAny>> {
    for p in profiles.try_iter()? {
        let p = p?;
        if p.getattr("id")?.extract::<String>()? == id {
            return Ok(p.unbind());
        }
    }
    Err(invalid(format!(
        "Remote Prompt Enhancer profile is not registered: {id}"
    )))
}
#[pyfunction]
fn profile_mutate_native(
    profiles: Vec<Py<PyAny>>,
    profile: Option<Py<PyAny>>,
    id: &str,
    py: Python<'_>,
) -> PyResult<(Vec<Py<PyAny>>, bool)> {
    let mut kept = Vec::new();
    let mut changed = false;
    for item in profiles {
        if item.bind(py).getattr("id")?.extract::<String>()? == id {
            changed = true;
        } else {
            kept.push(item);
        }
    }
    if let Some(profile) = profile {
        kept.push(profile);
        changed = true;
    }
    let mut sorted = Vec::new();
    for p in kept {
        sorted.push((p.bind(py).getattr("id")?.extract::<String>()?, p));
    }
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((sorted.into_iter().map(|(_, p)| p).collect(), changed))
}

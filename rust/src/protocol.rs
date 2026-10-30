use crate::{from_python, to_python};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use serde_json::{Value, json};

/// Validated identifier; construction is confined to the trust boundary.
///
/// ```compile_fail
/// use _native::RequestId;
/// let unchecked = RequestId("../other-request".to_owned());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestId(String);
impl RequestId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn parse(value: &str) -> Option<Self> {
        let bytes = value.as_bytes();
        if !(1..=64).contains(&bytes.len())
            || !bytes[0].is_ascii_alphanumeric()
            || !bytes
                .iter()
                .all(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
        {
            return None;
        }
        Some(Self(value.to_owned()))
    }
}
#[pyfunction]
fn is_valid_request_id(value: &Bound<'_, PyAny>) -> bool {
    value
        .extract::<String>()
        .ok()
        .is_some_and(|v| RequestId::parse(&v).is_some())
}
#[pyfunction]
fn validate_server_url(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let invalid = || {
        PyValueError::new_err("Remote Conditioning server_url must be an absolute HTTP(S) origin")
    };
    let raw = value.extract::<String>().map_err(|_| invalid())?;
    let parsed = url::Url::parse(&raw).map_err(|_| invalid())?;
    // Reject explicit empty userinfo and query/fragment delimiters as well.
    let authority = raw
        .split_once("://")
        .map(|(_, s)| s.split('/').next().unwrap_or(""))
        .unwrap_or("");
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || authority.contains('@')
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some_and(|s| !s.is_empty())
        || parsed.fragment().is_some_and(|s| !s.is_empty())
        || !matches!(parsed.path(), "" | "/")
    {
        return Err(invalid());
    }
    Ok(raw.trim_end_matches('/').to_owned())
}
#[pyfunction]
fn validate_policy(
    value: &Bound<'_, PyAny>,
    allowed: Vec<String>,
    field_name: &str,
) -> PyResult<String> {
    let s = value.extract::<String>().ok();
    if let Some(s) = s.filter(|s| allowed.contains(s)) {
        return Ok(s);
    }
    Err(PyValueError::new_err(format!(
        "Remote Conditioning {field_name} must be one of: {}",
        allowed.join(", ")
    )))
}
#[pyfunction]
pub(crate) fn validate_clip_values(values: &Bound<'_, PyAny>) -> PyResult<()> {
    let parts=values.extract::<Vec<String>>().map_err(|_|PyValueError::new_err("Remote Conditioning CLIP selection requires non-empty clip_name, clip_type, and device"))?;
    if parts.len() != 3 || parts.iter().any(|s| s.trim().is_empty()) {
        return Err(PyValueError::new_err(
            "Remote Conditioning CLIP selection requires non-empty clip_name, clip_type, and device",
        ));
    }
    if parts.iter().any(|s| s.contains('\0')) {
        return Err(PyValueError::new_err(
            "Remote Conditioning CLIP selection contains a null byte",
        ));
    }
    Ok(())
}
#[pyfunction]
fn resolve_peer_model_name(value: String, choices: Vec<String>, field: &str) -> PyResult<String> {
    if choices.contains(&value) {
        return Ok(value);
    }
    let normalized = value.replace('\\', "/");
    let matching: Vec<_> = choices
        .into_iter()
        .filter(|s| s.replace('\\', "/") == normalized)
        .collect();
    if matching.len() == 1 {
        return Ok(matching[0].clone());
    }
    let reason = if matching.is_empty() {
        "model unavailable"
    } else {
        "ambiguous model path"
    };
    Err(PyValueError::new_err(format!(
        "Remote peer {field}: {reason}: {value:?}; refresh the peer model list and select again"
    )))
}
fn choices(value: &Value, node: &str, name: &str) -> PyResult<Value> {
    let list = value
        .as_array()
        .filter(|v| {
            !v.is_empty()
                && v.iter()
                    .all(|s| s.as_str().is_some_and(|s| !s.trim().is_empty()))
        })
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Remote Conditioning {node} {name} options are invalid"
            ))
        })?;
    let mut unique = Vec::new();
    for v in list {
        if !unique.contains(v) {
            unique.push(v.clone());
        }
    }
    Ok(Value::Array(unique))
}
#[pyfunction]
fn parse_clip_loader_options(info: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = info.py();
    let info = from_python(info)?;
    let mut result = serde_json::Map::new();
    for (name, path) in [
        ("clip_name", "/CLIPLoader/input/required/clip_name/0"),
        ("clip_type", "/CLIPLoader/input/required/type/0"),
        ("device", "/CLIPLoader/input/optional/device/0"),
    ] {
        let value = info.pointer(path).ok_or_else(|| {
            PyRuntimeError::new_err("Remote Conditioning CLIPLoader object_info is malformed")
        })?;
        result.insert(name.to_owned(), choices(value, "CLIPLoader", name)?);
    }
    to_python(py, &Value::Object(result))
}
#[pyfunction]
fn parse_vae_loader_options(info: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let value = from_python(info)?;
    let values = value
        .pointer("/VAELoader/input/required/vae_name/0")
        .ok_or_else(|| {
            PyRuntimeError::new_err("Remote Conditioning VAELoader object_info is malformed")
        })?;
    to_python(
        info.py(),
        &json!({"vae_name":choices(values,"VAELoader","vae_name")?}),
    )
}
#[pyfunction]
fn build_release_cleanup_node(anything: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    to_python(
        anything.py(),
        &json!({"class_type":"VRAMCleanup","inputs":{"anything":from_python(anything)?,"offload_model":true,"offload_cache":true,"reset_execution_cache":false}}),
    )
}
#[pyfunction]
pub(crate) fn validate_cleanup_capability(info: &Bound<'_, PyAny>) -> PyResult<()> {
    let value = from_python(info)?;
    let inputs = value
        .get("input")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            PyRuntimeError::new_err(
                "Remote Conditioning release_after_result requires a valid VRAMCleanup capability",
            )
        })?;
    let required = inputs.get("required").ok_or_else(|| {
        PyRuntimeError::new_err(
            "Remote Conditioning release_after_result requires a valid VRAMCleanup capability",
        )
    })?;
    let optional = inputs.get("optional").ok_or_else(|| {
        PyRuntimeError::new_err(
            "Remote Conditioning release_after_result requires a valid VRAMCleanup capability",
        )
    })?;
    let (Some(required), Some(optional)) = (required.as_object(), optional.as_object()) else {
        return Err(PyRuntimeError::new_err(
            "Remote Conditioning VRAMCleanup capability has malformed inputs",
        ));
    };
    if !optional.contains_key("anything")
        || ["offload_model", "offload_cache"]
            .iter()
            .any(|k| !required.contains_key(*k) && !optional.contains_key(*k))
    {
        return Err(PyRuntimeError::new_err(
            "Remote Conditioning release_after_result requires VRAMCleanup inputs offload_model, offload_cache, and optional anything",
        ));
    }
    Ok(())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_valid_request_id, m)?)?;
    m.add_function(wrap_pyfunction!(validate_server_url, m)?)?;
    m.add_function(wrap_pyfunction!(validate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(validate_clip_values, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_peer_model_name, m)?)?;
    m.add_function(wrap_pyfunction!(parse_clip_loader_options, m)?)?;
    m.add_function(wrap_pyfunction!(parse_vae_loader_options, m)?)?;
    m.add_function(wrap_pyfunction!(build_release_cleanup_node, m)?)?;
    m.add_function(wrap_pyfunction!(validate_cleanup_capability, m)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ids_are_ascii_and_bounded() {
        for s in ["", "../bad", "a/b", "é", "a\0", "-a"] {
            assert!(RequestId::parse(s).is_none());
        }
        assert!(RequestId::parse(&"a".repeat(65)).is_none());
        assert!(RequestId::parse("a_1-x").is_some());
    }
}

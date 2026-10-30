use crate::protocol::RequestId;
use crate::{from_python, to_python};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use serde_json::{Map, Value};

fn complete<'a>(history: &'a Value, id: &str) -> PyResult<&'a Value> {
    let malformed = || {
        PyRuntimeError::new_err(format!(
            "Remote Conditioning history is malformed for prompt {id}"
        ))
    };
    let entry = history.get(id).ok_or_else(malformed)?;
    let status = entry.pointer("/status/status_str").ok_or_else(malformed)?;
    if status == "error" {
        if let Some(messages) = entry.pointer("/status/messages").and_then(Value::as_array) {
            for message in messages.iter().rev() {
                let Some(message) = message
                    .as_array()
                    .filter(|a| a.len() == 2 && a[0] == "execution_error")
                else {
                    continue;
                };
                let detail = &message[1];
                if let (Some(node), Some(kind), Some(text)) = (
                    detail.get("node_type").and_then(Value::as_str),
                    detail.get("exception_type").and_then(Value::as_str),
                    detail.get("exception_message").and_then(Value::as_str),
                ) {
                    if [node, kind, text].iter().all(|s| !s.trim().is_empty()) {
                        let concise: String = text
                            .split_whitespace()
                            .collect::<Vec<_>>()
                            .join(" ")
                            .chars()
                            .take(500)
                            .collect();
                        return Err(PyRuntimeError::new_err(format!(
                            "Remote Conditioning peer execution failed at {node}: {kind}: {concise} (prompt {id})"
                        )));
                    }
                }
            }
        }
        return Err(PyRuntimeError::new_err(format!(
            "Remote Conditioning peer execution failed for prompt {id}"
        )));
    }
    if status != "success" {
        return Err(PyRuntimeError::new_err(format!(
            "Remote Conditioning is not complete for prompt {id}: {status}"
        )));
    }
    Ok(entry)
}
#[pyfunction]
pub(crate) fn successful_outputs(
    history: &Bound<'_, PyAny>,
    prompt_id: &str,
) -> PyResult<Py<PyAny>> {
    let value = from_python(history)?;
    let entry = complete(&value, prompt_id)?;
    to_python(
        history.py(),
        entry.get("outputs").unwrap_or(&Value::Object(Map::new())),
    )
}
fn scoped(path: &str, request: &RequestId, extension: &str) -> bool {
    let request = request.as_str();
    path.starts_with(&format!("remote-conditioning/{request}/"))
        && !path.contains('\\')
        && !path.contains('\0')
        && path.split('/').all(|p| !matches!(p, "" | "." | ".."))
        && path.ends_with(extension)
}
#[pyfunction]
pub(crate) fn extract_artifact_paths(
    history: &Bound<'_, PyAny>,
    prompt_id: &str,
    request_id: &str,
    specs: Vec<(String, String, String, String)>,
) -> PyResult<Py<PyAny>> {
    let mut names = std::collections::HashSet::new();
    if specs.is_empty() || specs.iter().any(|s| !names.insert(s.0.clone())) {
        return Err(PyValueError::new_err(
            "Remote Conditioning artifact specs must be non-empty with unique names",
        ));
    }
    let request_id = RequestId::parse(request_id).ok_or_else(|| {
        PyValueError::new_err("Remote Conditioning request_id must be 1-64 URL-safe characters")
    })?;
    let value = from_python(history)?;
    let entry = complete(&value, prompt_id)?;
    let mut artifacts = Map::new();
    for (name, node, field, extension) in specs {
        let locators = entry
            .get("outputs")
            .and_then(|v| v.get(&node))
            .and_then(|v| v.get(&field))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Remote Conditioning {name} output is missing for prompt {prompt_id}"
                ))
            })?;
        let locators=locators.as_array().filter(|a|a.len()==1).ok_or_else(||PyRuntimeError::new_err(format!("Remote Conditioning must produce exactly one {name} artifact for prompt {prompt_id}")))?;
        let locator = &locators[0];
        let invalid = || {
            PyRuntimeError::new_err(format!(
                "Remote Conditioning {name} artifact locator is invalid"
            ))
        };
        let path = if let Some(path) = locator.as_str() {
            path.to_owned()
        } else {
            let filename = locator
                .get("filename")
                .and_then(Value::as_str)
                .ok_or_else(invalid)?;
            let folder = locator
                .get("subfolder")
                .and_then(Value::as_str)
                .ok_or_else(invalid)?;
            if locator.get("type") != Some(&Value::String("output".into())) {
                return Err(invalid());
            }
            if folder.is_empty() {
                filename.to_owned()
            } else {
                format!("{folder}/{filename}")
            }
        };
        if !scoped(&path, &request_id, &extension) {
            return Err(PyRuntimeError::new_err(format!(
                "Remote Conditioning {name} artifact is outside its request directory: {path:?}"
            )));
        }
        artifacts.insert(name, Value::String(path));
    }
    to_python(history.py(), &Value::Object(artifacts))
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(successful_outputs, m)?)?;
    m.add_function(wrap_pyfunction!(extract_artifact_paths, m)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scope_rejects_traversal() {
        for p in [
            "remote-conditioning/a/../x.cond",
            "remote-conditioning/b/x.cond",
            "remote-conditioning/a//x.cond",
            "remote-conditioning/a/x\\y.cond",
        ] {
            assert!(!scoped(p, &RequestId::parse("a").unwrap(), ".cond"));
        }
        assert!(scoped(
            "remote-conditioning/a/x.cond",
            &RequestId::parse("a").unwrap(),
            ".cond"
        ));
    }
}

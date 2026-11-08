use super::{clean_model_name_native, invalid, minimax_policy_native};
use crate::from_python;
use pyo3::{exceptions::*, prelude::*, types::*};
use serde_json::Value;
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

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resolve_vae_name_native, m)?)?;
    m.add_function(wrap_pyfunction!(add_release_cleanup_native, m)?)?;
    Ok(())
}

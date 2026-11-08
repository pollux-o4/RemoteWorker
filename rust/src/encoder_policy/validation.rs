use pyo3::{exceptions::*, prelude::*, types::*};
pub(super) fn invalid(s: impl Into<String>) -> PyErr {
    PyValueError::new_err(s.into())
}
#[pyfunction]
pub(super) fn clean_model_name_native(
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
pub(super) fn minimax_policy_native(value: &Bound<'_, PyAny>, cache: bool) -> PyResult<String> {
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
pub(crate) fn validate_prompt_media_native(
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

pub(super) fn register_base(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clean_model_name_native, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_policy_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_connection_native, m)?)?;
    Ok(())
}
pub(super) fn register_qwen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qwen_worker_inputs_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_resize_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_prompt_media_native, m)?)?;
    Ok(())
}
pub(super) fn register_local(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encoder_local_inputs_native, m)?)?;
    m.add_function(wrap_pyfunction!(release_capability_native, m)?)?;
    Ok(())
}

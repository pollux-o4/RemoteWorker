use super::invalid;
use pyo3::{prelude::*, types::*};
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

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cache_identity_native, m)?)?;
    m.add_function(wrap_pyfunction!(cache_fingerprint_native, m)?)?;
    Ok(())
}

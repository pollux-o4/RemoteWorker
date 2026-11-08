use super::invalid;
use pyo3::{prelude::*, types::*};
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
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connected_slots_native, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_images_native, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_audio_native, m)?)?;
    m.add_function(wrap_pyfunction!(single_frame_native, m)?)?;
    m.add_function(wrap_pyfunction!(qwen_socket_count_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_qwen_frame_count_native, m)?)?;
    Ok(())
}

use pyo3::{prelude::*, types::*};
use std::collections::HashMap;
fn digest(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    py.import("hashlib")?
        .call_method1("sha256", (value,))?
        .call_method0("hexdigest")?
        .extract()
}
fn signature_value(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    if value.hasattr("detach")? {
        let tensor = value
            .call_method0("detach")?
            .call_method0("cpu")?
            .call_method0("contiguous")?;
        let bytes = tensor
            .call_method1("view", (py.import("torch")?.getattr("uint8")?,))?
            .call_method0("numpy")?
            .call_method0("tobytes")?;
        let result = PyDict::new(py);
        result.set_item(
            "shape",
            py.import("builtins")?
                .getattr("list")?
                .call1((tensor.getattr("shape")?,))?,
        )?;
        result.set_item("dtype", tensor.getattr("dtype")?.str()?)?;
        result.set_item("sha256", digest(py, &bytes)?)?;
        return Ok(result.into_any().unbind());
    }
    if let Ok(values) = value.cast::<PyDict>() {
        let result = PyDict::new(py);
        for (k, v) in values.iter() {
            result.set_item(k, signature_value(&v)?)?;
        }
        return Ok(result.into_any().unbind());
    }
    if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
        let result = PyList::empty(py);
        for v in value.try_iter()? {
            result.append(signature_value(&v?)?)?;
        }
        return Ok(result.into_any().unbind());
    }
    Ok(value.clone().unbind())
}
#[pyfunction]
pub(super) fn handoff_signature(values: &Bound<'_, PyDict>) -> PyResult<String> {
    let py = values.py();
    let filtered = PyDict::new(py);
    for (k, v) in values.iter() {
        let name: String = k.extract()?;
        if !["self", "unique_id", "extra_pnginfo", "timeout_seconds"].contains(&name.as_str()) {
            filtered.set_item(k, v)?;
        }
    }
    let normalized = signature_value(filtered.as_any())?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("sort_keys", true)?;
    let encoded = py
        .import("json")?
        .call_method("dumps", (normalized,), Some(&kwargs))?
        .call_method0("encode")?;
    digest(py, &encoded)
}
fn tensor_bytes(value: &Bound<'_, PyAny>) -> PyResult<u128> {
    if value.hasattr("numel")? {
        if value
            .getattr("is_nested")
            .and_then(|v| v.is_truthy())
            .unwrap_or(false)
        {
            let mut bytes = 0;
            for v in value.call_method0("unbind")?.try_iter()? {
                bytes += tensor_bytes(&v?)?;
            }
            return Ok(bytes);
        }
        let elements: u128 = value.call_method0("numel")?.extract()?;
        let size: u128 = value.call_method0("element_size")?.extract()?;
        return Ok(elements * size);
    }
    let mut size = 0;
    if let Ok(values) = value.cast::<PyDict>() {
        for (_, v) in values.iter() {
            size += tensor_bytes(&v)?;
        }
    } else if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
        for v in value.try_iter()? {
            size += tensor_bytes(&v?)?;
        }
    }
    Ok(size)
}
#[pyfunction]
#[pyo3(signature=(result,max_bytes=None))]
pub(super) fn handoff_pack(
    result: &Bound<'_, PyAny>,
    max_bytes: Option<u128>,
) -> PyResult<Option<Py<PyAny>>> {
    let py = result.py();
    if let Some(max) = max_bytes {
        if tensor_bytes(result)? > max {
            return Ok(None);
        }
    }
    let buffer = py.import("io")?.getattr("BytesIO")?.call0()?;
    py.import("torch")?
        .call_method1("save", (result, &buffer))?;
    let size: u128 = buffer.call_method0("tell")?.extract()?;
    if max_bytes.is_some_and(|max| size > max) {
        return Ok(None);
    }
    Ok(Some(buffer.call_method0("getvalue")?.unbind()))
}
#[pyfunction]
pub(super) fn handoff_unpack(blob: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = blob.py();
    let buffer = py.import("io")?.getattr("BytesIO")?.call1((blob,))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("map_location", "cpu")?;
    kwargs.set_item("weights_only", false)?;
    Ok(py
        .import("torch")?
        .call_method("load", (buffer,), Some(&kwargs))?
        .unbind())
}
#[pyfunction]
pub(super) fn memory_snapshot(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return Ok(result.unbind());
    };
    let values: HashMap<_, _> = text
        .lines()
        .filter_map(|line| {
            let (name, raw) = line.split_once(':')?;
            Some((
                name,
                raw.split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)?,
            ))
        })
        .collect();
    if let (Some(available), Some(swap), Some(free)) = (
        values.get("MemAvailable"),
        values.get("SwapTotal"),
        values.get("SwapFree"),
    ) {
        result.set_item("available", available)?;
        result.set_item("swap_used", swap.saturating_sub(*free))?;
    }
    Ok(result.unbind())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(handoff_signature, m)?)?;
    m.add_function(wrap_pyfunction!(handoff_pack, m)?)?;
    m.add_function(wrap_pyfunction!(handoff_unpack, m)?)?;
    m.add_function(wrap_pyfunction!(memory_snapshot, m)?)?;
    Ok(())
}

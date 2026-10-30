use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};

fn tensor(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value
        .py()
        .import("torch")?
        .call_method1("is_tensor", (value,))?
        .is_truthy()
}
fn nested(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value
        .getattr("is_nested")
        .map(|v| v.is_truthy())
        .unwrap_or(Ok(false))
}
pub fn cpu(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    if nested(value)? {
        let result = PyDict::new(py);
        let parts = PyList::empty(py);
        for part in value.call_method0("unbind")?.try_iter()? {
            parts.append(cpu(&part?)?)?;
        }
        result.set_item("__remote_vae_nested__", parts)?;
        return Ok(result.into_any().unbind());
    }
    if tensor(value)? {
        return Ok(value.call_method0("detach")?.call_method0("cpu")?.unbind());
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        if dict.contains("__remote_vae_nested__")? {
            return Err(PyValueError::new_err("Reserved Remote VAE envelope key"));
        }
        let result = PyDict::new(py);
        for (k, v) in dict.iter() {
            result.set_item(k, cpu(&v)?)?;
        }
        return Ok(result.into_any().unbind());
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let parts = value
            .try_iter()?
            .map(|p| cpu(&p?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(value.get_type().call1((PyList::new(py, parts)?,))?.unbind());
    }
    if value.is_none()
        || value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyBool>()
    {
        return Ok(value.clone().unbind());
    }
    Err(PyValueError::new_err(
        "Unsupported Remote VAE metadata type",
    ))
}
pub fn restore(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = value.py();
    if let Ok(dict) = value.cast::<PyDict>() {
        if dict.len() == 1 && dict.contains("__remote_vae_nested__")? {
            let parts = dict.get_item("__remote_vae_nested__")?.unwrap();
            let valid = if let Ok(parts) = parts.cast::<PyList>() {
                !parts.is_empty()
                    && parts
                        .iter()
                        .map(|v| tensor(&v))
                        .collect::<PyResult<Vec<_>>>()?
                        .iter()
                        .all(|v| *v)
            } else {
                false
            };
            if !valid {
                return Err(PyValueError::new_err("Invalid nested latent envelope"));
            }
            return Ok(py
                .import("comfy.nested_tensor")?
                .getattr("NestedTensor")?
                .call1((parts,))?
                .unbind());
        }
        let result = PyDict::new(py);
        for (k, v) in dict.iter() {
            result.set_item(k, restore(&v)?)?;
        }
        return Ok(result.into_any().unbind());
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let parts = value
            .try_iter()?
            .map(|p| restore(&p?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(value.get_type().call1((PyList::new(py, parts)?,))?.unbind());
    }
    Ok(value.clone().unbind())
}
fn tensor_bytes(value: &Bound<'_, PyAny>, limit: usize) -> PyResult<usize> {
    if tensor(value)? {
        if nested(value)? {
            let mut size = 0usize;
            for part in value.call_method0("unbind")?.try_iter()? {
                size = size.saturating_add(tensor_bytes(&part?, limit)?);
            }
            return Ok(size);
        }
        let count = value.call_method0("numel")?.extract::<usize>()?;
        let element = value.call_method0("element_size")?.extract::<usize>()?;
        return Ok(count.saturating_mul(element));
    }
    let mut size = 0usize;
    if let Ok(dict) = value.cast::<PyDict>() {
        for (_, v) in dict.iter() {
            size = size.saturating_add(tensor_bytes(&v, limit)?);
            if size > limit {
                break;
            }
        }
    } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        for v in value.try_iter()? {
            size = size.saturating_add(tensor_bytes(&v?, limit)?);
            if size > limit {
                break;
            }
        }
    }
    Ok(size)
}
#[pyfunction]
fn vae_cpu_value(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    cpu(value)
}
#[pyfunction]
fn vae_restore(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    restore(value)
}
#[pyfunction]
pub(super) fn vae_dumps(value: &Bound<'_, PyAny>, max_bytes: usize) -> PyResult<Py<PyBytes>> {
    let py = value.py();
    let value = cpu(value)?;
    if tensor_bytes(value.bind(py), max_bytes)? > max_bytes {
        return Err(PyValueError::new_err(
            "Remote VAE tensor exceeds the 512 MiB transfer limit",
        ));
    }
    let stream = py.import("io")?.getattr("BytesIO")?.call0()?;
    py.import("torch")?.call_method1("save", (value, &stream))?;
    let bytes = stream.call_method0("getvalue")?.cast_into::<PyBytes>()?;
    if bytes.as_bytes().len() > max_bytes {
        return Err(PyValueError::new_err(
            "Remote VAE envelope exceeds the 512 MiB transfer limit",
        ));
    }
    Ok(bytes.unbind())
}
#[pyfunction]
pub(super) fn vae_loads(data: &Bound<'_, PyBytes>, max_bytes: usize) -> PyResult<Py<PyAny>> {
    if data.as_bytes().is_empty() || data.as_bytes().len() > max_bytes {
        return Err(PyValueError::new_err("Remote VAE envelope size is invalid"));
    }
    let py = data.py();
    let stream = py.import("io")?.getattr("BytesIO")?.call1((data,))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("map_location", "cpu")?;
    kwargs.set_item("weights_only", true)?;
    restore(
        &py.import("torch")?
            .call_method("load", (stream,), Some(&kwargs))?,
    )
}
pub fn latent(value: &Bound<'_, PyAny>) -> PyResult<()> {
    let valid = if let Ok(dict) = value.cast::<PyDict>() {
        if let Some(samples) = dict.get_item("samples")? {
            tensor(&samples)? || nested(&samples)?
        } else {
            false
        }
    } else {
        false
    };
    if valid {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "Remote VAE requires a LATENT dictionary with tensor samples",
        ))
    }
}
#[pyfunction]
fn vae_validate_latent(value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    latent(value)?;
    Ok(value.clone().unbind())
}
#[pyfunction]
pub(super) fn vae_validate_result(value: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
    let valid = if kind == "image" {
        tensor(value)?
            && value.getattr("ndim")?.extract::<usize>()? == 4
            && [3, 4].contains(&value.getattr("shape")?.get_item(-1)?.extract::<usize>()?)
    } else {
        if let Ok(dict) = value.cast::<PyDict>() {
            match (dict.get_item("waveform")?, dict.get_item("sample_rate")?) {
                (Some(wave), Some(rate)) => {
                    tensor(&wave)?
                        && wave.getattr("ndim")?.extract::<usize>()? == 3
                        && rate.is_instance_of::<PyInt>()
                        && !rate.is_instance_of::<PyBool>()
                        && rate.extract::<i64>()? > 0
                }
                _ => false,
            }
        } else {
            false
        }
    };
    if !valid {
        return Err(PyValueError::new_err(format!(
            "Remote VAE returned invalid {kind} data"
        )));
    }
    Ok(value.clone().unbind())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vae_cpu_value, m)?)?;
    m.add_function(wrap_pyfunction!(vae_restore, m)?)?;
    m.add_function(wrap_pyfunction!(vae_dumps, m)?)?;
    m.add_function(wrap_pyfunction!(vae_loads, m)?)?;
    m.add_function(wrap_pyfunction!(vae_validate_latent, m)?)?;
    m.add_function(wrap_pyfunction!(vae_validate_result, m)?)?;
    Ok(())
}

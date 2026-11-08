use pyo3::{prelude::*, types::*};
use std::collections::VecDeque;
enum WeakValue {
    Primitive(Py<PyAny>),
    Reference(Py<PyAny>),
    List(Vec<WeakValue>),
    Tuple(Vec<WeakValue>),
}
impl WeakValue {
    fn pack(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            let mut parts = Vec::new();
            for part in value.try_iter()? {
                parts.push(Self::pack(&part?)?);
            }
            return Ok(if value.is_instance_of::<PyList>() {
                Self::List(parts)
            } else {
                Self::Tuple(parts)
            });
        }
        if value.is_none()
            || value.is_instance_of::<PyString>()
            || value.is_instance_of::<PyInt>()
            || value.is_instance_of::<PyFloat>()
        {
            return Ok(Self::Primitive(value.clone().unbind()));
        }
        Ok(Self::Reference(
            value
                .py()
                .import("weakref")?
                .getattr("ref")?
                .call1((value,))?
                .unbind(),
        ))
    }
    fn unpack(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match self {
            Self::Primitive(value) => Ok(Some(value.clone_ref(py))),
            Self::Reference(reference) => {
                let value = reference.call0(py)?;
                Ok(if value.is_none(py) { None } else { Some(value) })
            }
            Self::List(parts) | Self::Tuple(parts) => {
                let mut result = Vec::new();
                for part in parts {
                    let Some(value) = part.unpack(py)? else {
                        return Ok(None);
                    };
                    result.push(value);
                }
                Ok(Some(match self {
                    Self::List(_) => PyList::new(py, result)?.into_any().unbind(),
                    _ => PyTuple::new(py, result)?.into_any().unbind(),
                }))
            }
        }
    }
}
#[pyclass]
struct WeakModelCache {
    entries: VecDeque<(Py<PyAny>, WeakValue)>,
    #[pyo3(get, set)]
    hits: u64,
}
#[pymethods]
impl WeakModelCache {
    #[new]
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            hits: 0,
        }
    }
    fn clear(&mut self) {
        self.entries.clear();
    }
    fn remember(&mut self, key: &Bound<'_, PyAny>, outputs: &Bound<'_, PyAny>) -> PyResult<()> {
        let packed = match WeakValue::pack(outputs) {
            Ok(value) => value,
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyTypeError>(key.py()) => {
                return Ok(());
            }
            Err(err) => return Err(err),
        };
        if let Some(index) = self.index(key)? {
            self.entries.remove(index);
        }
        self.entries.push_back((key.clone().unbind(), packed));
        while self.entries.len() > 16 {
            self.entries.pop_front();
        }
        Ok(())
    }
    fn recall(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let Some(index) = self.index(key)? else {
            return Ok(None);
        };
        let result = self.entries[index].1.unpack(key.py())?;
        if result.is_none() {
            self.entries.remove(index);
        }
        Ok(result)
    }
}
impl WeakModelCache {
    fn index(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        key.hash()?;
        for (index, (stored, _)) in self.entries.iter().enumerate() {
            if stored.bind(key.py()).eq(key)? {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }
}
#[pyfunction]
#[pyo3(signature=(cls,value=None))]
fn model_cache_eligible(
    cls: &Bound<'_, PyAny>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    if cls.is_none() {
        return Ok(false);
    }
    let returns = cls
        .getattr("RETURN_TYPES")
        .unwrap_or_else(|_| PyTuple::empty(cls.py()).into_any());
    if !returns.contains("MODEL")? {
        return Ok(false);
    }
    Ok(!value.is_some_and(|v| {
        v.is_truthy().unwrap_or(false)
            && v.getattr("ui").and_then(|v| v.is_truthy()).unwrap_or(false)
    }))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WeakModelCache>()?;
    m.add_function(wrap_pyfunction!(model_cache_eligible, m)?)?;
    Ok(())
}

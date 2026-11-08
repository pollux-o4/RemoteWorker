pyo3::create_exception!(_native, UnsupportedInput, PyValueError);
const SAFE_TYPES: &[&str] = &[
    "LoadImage",
    "ResizeImageMaskNode",
    "ResolutionSelector",
    "GetImageSize",
    "ImageScaleToTotalPixels",
    "ImageScale",
    "ImageScaleBy",
    "ImageFromBatch",
    "ImageBatch",
    "PrimitiveFloat",
    "PrimitiveInt",
    "PrimitiveBoolean",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "RequestConditionInput",
    "GenRelInput",
    "ComfyMathExpression",
    "RemoteVAESelector",
];

use pyo3::{exceptions::*, prelude::*, types::*};
use std::collections::HashSet;
#[pyclass]
pub(super) struct Resolver {
    graph: Py<PyDict>,
    classes: Py<PyDict>,
    outputs: Py<PyDict>,
    visiting: HashSet<String>,
    check_cancelled: Py<PyAny>,
}
#[pymethods]
impl Resolver {
    #[new]
    pub(super) fn new(
        py: Python<'_>,
        graph: Py<PyDict>,
        classes: Py<PyDict>,
        check_cancelled: Py<PyAny>,
    ) -> Self {
        Self {
            graph,
            classes,
            outputs: PyDict::new(py).unbind(),
            visiting: HashSet::new(),
            check_cancelled,
        }
    }
    #[pyo3(signature=(node_id,exclude=None))]
    fn inputs(
        &mut self,
        py: Python<'_>,
        node_id: &str,
        exclude: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        self.resolve_inputs(py, node_id, exclude)
    }
    fn value(&mut self, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.resolve_value(value)
    }
}
fn required<'py>(values: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    values
        .get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))
}
impl Resolver {
    pub(super) fn resolve_inputs(
        &mut self,
        py: Python<'_>,
        node_id: &str,
        exclude: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        let node = self
            .graph
            .bind(py)
            .get_item(node_id)?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(node_id.to_owned()))?;
        let inputs = node
            .call_method1("get", ("inputs", PyDict::new(py)))?
            .cast_into::<PyDict>()?;
        let result = PyDict::new(py);
        for (key, value) in inputs.iter() {
            if exclude.is_some_and(|exclude| exclude.contains(&key).unwrap_or(false)) {
                continue;
            }
            result.set_item(key, self.resolve_value(&value)?)?;
        }
        Ok(result.unbind())
    }
    fn resolve_value(&mut self, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = value.py();
        let Ok(link) = value.cast::<PyList>() else {
            return Ok(value.clone().unbind());
        };
        if link.len() != 2 {
            return Ok(value.clone().unbind());
        }
        let key: String = link.get_item(0)?.str()?.extract()?;
        let Some(node) = self.graph.bind(py).get_item(&key)? else {
            return Ok(value.clone().unbind());
        };
        let slot = link.get_item(1)?;
        if !self.outputs.bind(py).contains(&key)? {
            let kind: String = node.get_item("class_type")?.extract()?;
            let cls = self.classes.bind(py).get_item(&kind)?;
            if !SAFE_TYPES.contains(&kind.as_str()) || cls.is_none() {
                return Err(UnsupportedInput::new_err(format!(
                    "upstream_not_cpu_safe: {kind}"
                )));
            }
            if self.visiting.contains(&key) || self.visiting.len() > 64 {
                return Err(UnsupportedInput::new_err("upstream_cycle_or_depth"));
            }
            self.visiting.insert(key.clone());
            let result = self.execute(py, &key, &kind, cls.unwrap());
            self.visiting.remove(&key);
            result?;
        }
        Ok(self
            .outputs
            .bind(py)
            .get_item(&key)?
            .unwrap()
            .get_item(slot)?
            .unbind())
    }
    fn execute(
        &mut self,
        py: Python<'_>,
        key: &str,
        kind: &str,
        cls: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.check_cancelled.call0(py)?;
        let mut values = self.resolve_inputs(py, key, None)?;
        if kind == "ComfyMathExpression" {
            let mapped = PyDict::new(py);
            mapped.set_item("expression", required(values.bind(py), "expression")?)?;
            let nested = PyDict::new(py);
            for (k, v) in values.bind(py).iter() {
                let k: String = k.extract()?;
                if let Some(k) = k.strip_prefix("values.") {
                    nested.set_item(k, v)?;
                }
            }
            mapped.set_item("values", nested)?;
            values = mapped.unbind();
        }
        if kind == "ResizeImageMaskNode" {
            let mapped = PyDict::new(py);
            let nested = PyDict::new(py);
            for field in ["input", "scale_method"] {
                mapped.set_item(field, required(values.bind(py), field)?)?;
            }
            nested.set_item("resize_type", required(values.bind(py), "resize_type")?)?;
            for (k, v) in values.bind(py).iter() {
                let k: String = k.extract()?;
                if let Some(k) = k.strip_prefix("resize_type.") {
                    nested.set_item(k, v)?;
                }
            }
            mapped.set_item("resize_type", nested)?;
            values = mapped.unbind();
        }
        let mut result = if kind == "GetImageSize" {
            let shape = required(values.bind(py), "image")?.getattr("shape")?;
            PyTuple::new(
                py,
                [shape.get_item(2)?, shape.get_item(1)?, shape.get_item(0)?],
            )?
            .into_any()
        } else {
            let function = cls
                .getattr("FUNCTION")
                .map(|s| s.extract::<String>())
                .unwrap_or_else(|_| Ok("execute".into()))?;
            cls.call0()?
                .getattr(function.as_str())?
                .call((), Some(values.bind(py)))?
        };
        if result.hasattr("result")? {
            result = result.getattr("result")?;
        }
        if let Ok(mapping) = result.cast::<PyDict>() {
            if mapping
                .get_item("expand")?
                .is_some_and(|v| v.is_truthy().unwrap_or(false))
            {
                return Err(UnsupportedInput::new_err("upstream_expansion"));
            }
            result = mapping
                .get_item("result")?
                .unwrap_or_else(|| py.None().into_bound(py));
        }
        if !result.is_instance_of::<PyTuple>() && !result.is_instance_of::<PyList>() {
            return Err(UnsupportedInput::new_err(format!(
                "upstream_result: {kind}"
            )));
        }
        self.outputs.bind(py).set_item(key, result)?;
        self.check_cancelled.call0(py)?;
        Ok(())
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Resolver>()?;
    m.add("UnsupportedInput", m.py().get_type::<UnsupportedInput>())?;
    Ok(())
}

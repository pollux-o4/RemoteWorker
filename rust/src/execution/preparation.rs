use super::slots::validate_preparation;
use pyo3::{prelude::*, types::*};
use std::collections::{HashMap, HashSet};
#[derive(Clone)]
enum PreparationRole {
    Encoder(Option<String>),
    ImageDecoder,
    AudioDecoder,
}
#[pyclass(frozen)]
struct Preparation {
    role: PreparationRole,
    #[pyo3(get)]
    executor: Option<Py<PyAny>>,
    #[pyo3(get)]
    inputs: Py<PyAny>,
}
#[pymethods]
impl Preparation {
    #[new]
    #[pyo3(signature=(role,kind=None,executor=None,inputs=None))]
    fn new(
        py: Python<'_>,
        role: &str,
        kind: Option<String>,
        executor: Option<Py<PyAny>>,
        inputs: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        validate_preparation(role, kind.as_deref())?;
        let role = match role {
            "encoder" => PreparationRole::Encoder(kind),
            "decoder" if kind.as_deref() == Some("image") => PreparationRole::ImageDecoder,
            _ => PreparationRole::AudioDecoder,
        };
        Ok(Self {
            role,
            executor,
            inputs: match inputs {
                Some(v) => v,
                None => py.import("builtins")?.getattr("dict")?.unbind(),
            },
        })
    }
    #[getter]
    fn role(&self) -> &str {
        match self.role {
            PreparationRole::Encoder(_) => "encoder",
            _ => "decoder",
        }
    }
    #[getter]
    fn kind(&self) -> Option<String> {
        match &self.role {
            PreparationRole::Encoder(kind) => kind.clone(),
            PreparationRole::ImageDecoder => Some("image".into()),
            PreparationRole::AudioDecoder => Some("audio".into()),
        }
    }
}
#[pyfunction]
fn discover_preparation(classes: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(classes.py());
    for (name, cls) in classes.iter() {
        if let Ok(contract) = cls.getattr("REMOTE_PREPARATION") {
            if contract.is_instance_of::<Preparation>() {
                result.set_item(name, (cls, contract))?;
            }
        }
    }
    Ok(result.unbind())
}
#[pyfunction]
fn preparation_encoders(registered: &Bound<'_, PyDict>) -> PyResult<HashSet<String>> {
    let mut result = HashSet::new();
    for (name, value) in registered.iter() {
        let spec = value.get_item(1)?.extract::<PyRef<'_, Preparation>>()?;
        if matches!(spec.role, PreparationRole::Encoder(_)) {
            result.insert(name.extract()?);
        }
    }
    Ok(result)
}
#[pyfunction]
fn preparation_decoders(registered: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut result = HashMap::new();
    for (name, value) in registered.iter() {
        let spec = value.get_item(1)?.extract::<PyRef<'_, Preparation>>()?;
        if !matches!(spec.role, PreparationRole::Encoder(_)) {
            result.insert(name.extract()?, spec.kind().unwrap());
        }
    }
    Ok(result)
}
#[pyfunction]
fn sampling_in(registry: &Bound<'_, PyAny>, prompt_id: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !registry.getattr("prompt_id")?.eq(prompt_id)? {
        return Ok(false);
    }
    for (node_id, entry) in registry
        .getattr("nodes")?
        .call_method0("copy")?
        .cast::<PyDict>()?
        .iter()
    {
        let state = entry.call_method1("get", ("state",))?;
        let state = state.getattr("value").unwrap_or(state);
        if state.extract::<String>().is_ok_and(|s| s == "running") {
            let node = registry
                .getattr("dynprompt")?
                .call_method1("get_node", (node_id,))?;
            if node
                .call_method1("get", ("class_type", ""))?
                .extract::<String>()?
                .contains("Sampler")
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
#[pyfunction]
fn prep_observe_event(
    coordinator: &Bound<'_, PyAny>,
    event: &str,
    data: &Bound<'_, PyAny>,
    read_registry: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if event == "progress_state" && data.is_instance_of::<PyDict>() {
        let registry = read_registry.call0()?;
        let job = data.call_method1("get", ("prompt_id",))?;
        if registry.getattr("prompt_id")?.eq(&job)? {
            for (key, value) in data
                .call_method1("get", ("nodes", PyDict::new(data.py())))?
                .cast::<PyDict>()?
                .iter()
            {
                if value
                    .call_method1("get", ("state",))?
                    .extract::<String>()
                    .unwrap_or_default()
                    != "running"
                {
                    continue;
                }
                let node = registry
                    .getattr("dynprompt")?
                    .call_method1("get_node", (&key,))?;
                if node
                    .call_method1("get", ("class_type", ""))?
                    .extract::<String>()?
                    .contains("Sampler")
                {
                    coordinator
                        .getattr("scheduler")?
                        .getattr("sampling")?
                        .call_method1(
                            "update",
                            (
                                &job,
                                key,
                                value.call_method1("get", ("value", 0))?,
                                value.call_method1("get", ("max", 1))?,
                            ),
                        )?;
                }
            }
        }
        coordinator.getattr("wake")?.call_method0("set")?;
    } else if [
        "status",
        "executing",
        "execution_start",
        "execution_success",
        "execution_error",
        "execution_interrupted",
    ]
    .contains(&event)
    {
        coordinator.getattr("wake")?.call_method0("set")?;
    }
    Ok(())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Preparation>()?;
    m.add_function(wrap_pyfunction!(discover_preparation, m)?)?;
    m.add_function(wrap_pyfunction!(preparation_encoders, m)?)?;
    m.add_function(wrap_pyfunction!(preparation_decoders, m)?)?;
    m.add_function(wrap_pyfunction!(sampling_in, m)?)?;
    m.add_function(wrap_pyfunction!(prep_observe_event, m)?)?;
    Ok(())
}

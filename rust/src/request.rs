//! Targeted request lifetime: cancellation spans wait and result consumption.
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};
fn warn(py: Python<'_>, format: &str, args: impl IntoIterator<Item = Py<PyAny>>) {
    let result = (|| -> PyResult<()> {
        let log = py
            .import("logging")?
            .call_method1("getLogger", ("remoteworker.core.execution",))?;
        let mut values = vec![format.into_pyobject(py)?.into_any().unbind()];
        values.extend(args);
        log.call_method1("warning", PyTuple::new(py, values)?)?;
        Ok(())
    })();
    let _ = result;
}
fn cancel(api: &Bound<'_, PyAny>, prompt_id: &str) {
    if let Err(error) = api.call_method1("cancel_prompt", (prompt_id,)) {
        let py = api.py();
        warn(
            py,
            "RemoteWorker could not cancel remote prompt %s: %s",
            [
                prompt_id.into_pyobject(py).unwrap().into_any().unbind(),
                error.value(py).clone().into_any().unbind(),
            ],
        );
    }
}
#[pyclass]
struct RemoteFailureScope {
    api: Py<PyAny>,
    prompt_id: String,
}
#[pymethods]
impl RemoteFailureScope {
    #[new]
    fn new(api: Py<PyAny>, prompt_id: String) -> Self {
        Self { api, prompt_id }
    }
    fn __enter__(&self) {}
    fn __exit__(
        &self,
        py: Python<'_>,
        ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        if !ty.is_none() {
            cancel(self.api.bind(py), &self.prompt_id);
        }
        false
    }
}
#[pyclass]
struct CompletedPrompt {
    api: Py<PyAny>,
    workflow: Py<PyAny>,
    prompt_id: Option<String>,
}
#[pymethods]
impl CompletedPrompt {
    #[new]
    fn new(api: Py<PyAny>, workflow: Py<PyAny>) -> Self {
        Self {
            api,
            workflow,
            prompt_id: None,
        }
    }
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let api = self.api.bind(py);
        let id = api
            .call_method1("submit", (self.workflow.bind(py),))?
            .extract::<String>()
            .ok()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| PyRuntimeError::new_err("RemoteWorker API returned no prompt_id"))?;
        self.prompt_id = Some(id.clone());
        match api.call_method1("wait_for_history", (&id,)) {
            Ok(history) => Ok((id, history).into_pyobject(py)?.into_any().unbind()),
            Err(error) => {
                cancel(api, &id);
                Err(error)
            }
        }
    }
    fn __exit__(
        &self,
        py: Python<'_>,
        ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        if !ty.is_none() {
            if let Some(id) = &self.prompt_id {
                cancel(self.api.bind(py), id);
            }
        }
        false
    }
}
#[pyclass]
struct RequestInputs {
    #[pyo3(get)]
    api: Py<PyAny>,
    #[pyo3(get)]
    request_id: Py<PyAny>,
    #[pyo3(get)]
    uploaded: bool,
}
#[pymethods]
impl RequestInputs {
    #[new]
    fn new(api: Py<PyAny>, request_id: Py<PyAny>) -> Self {
        Self {
            api,
            request_id,
            uploaded: false,
        }
    }
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    #[pyo3(signature=(**kwargs))]
    fn upload_image(
        &mut self,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.upload(py, "upload_image", kwargs)
    }
    #[pyo3(signature=(**kwargs))]
    fn upload_file(
        &mut self,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.upload(py, "upload_file", kwargs)
    }
    fn __exit__(
        &self,
        py: Python<'_>,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if self.uploaded {
            if let Err(error) = self
                .api
                .bind(py)
                .call_method1("discard_inputs", (self.request_id.bind(py),))
            {
                if error.is_instance_of::<PyOSError>(py)
                    || error.is_instance_of::<PyRuntimeError>(py)
                    || error.is_instance_of::<PyTimeoutError>(py)
                    || error.is_instance_of::<PyValueError>(py)
                {
                    warn(
                        py,
                        "RemoteWorker could not discard request inputs: %s",
                        [error.value(py).clone().into_any().unbind()],
                    );
                } else {
                    return Err(error);
                }
            }
        }
        Ok(false)
    }
}
impl RequestInputs {
    fn upload(
        &mut self,
        py: Python<'_>,
        method: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let kwargs = kwargs
            .map(Bound::copy)
            .transpose()?
            .unwrap_or_else(|| PyDict::new(py));
        kwargs.set_item("request_id", self.request_id.bind(py))?;
        let result = self.api.bind(py).getattr(method)?.call((), Some(&kwargs))?;
        self.uploaded = true;
        Ok(result.unbind())
    }
}
#[pyfunction]
fn download_artifacts(api: &Bound<'_, PyAny>, paths: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(api.py());
    for (name, path) in paths.iter() {
        out.set_item(name, api.call_method1("download_artifact", (path,))?)?;
    }
    Ok(out.unbind())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RemoteFailureScope>()?;
    m.add_class::<CompletedPrompt>()?;
    m.add_class::<RequestInputs>()?;
    m.add_function(wrap_pyfunction!(download_artifacts, m)?)?;
    Ok(())
}

mod common;
mod conditioning;
mod minimax;
mod prompt;
mod qwen;

use common::*;
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
    types::{PyDict, PyModule},
};
#[pyfunction]
fn encoder_run_native(
    kind: &str,
    module: &Bound<'_, PyAny>,
    values: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let result = match kind {
        "qwen" => qwen::qwen_run(module, values),
        "prompt" => prompt::prompt_run(module, values),
        "conditioning" => conditioning::conditioning_run(module, values),
        "minimax-reference" => minimax::minimax_run(module, values, true),
        "minimax-i2v" => minimax::minimax_run(module, values, false),
        _ => return Err(PyValueError::new_err("Unknown remote encoder kind")),
    };
    match result {
        Ok(result) => Ok(result.unbind()),
        Err(error) => {
            let py = module.py();
            if kind != "qwen"
                && (error.is_instance_of::<PyOSError>(py)
                    || error.is_instance_of::<PyRuntimeError>(py)
                    || error.is_instance_of::<PyTimeoutError>(py)
                    || error.is_instance_of::<PyValueError>(py))
            {
                let prefix = match kind {
                    "conditioning" => "Remote Conditioning",
                    "prompt" => "Remote Prompt Enhancer",
                    "minimax-reference" => "Remote MiniMax H3",
                    _ => "Remote MiniMax H3 Image to Video",
                };
                let replacement =
                    PyRuntimeError::new_err(format!("{prefix} failed: {}", error.value(py).str()?));
                replacement.set_cause(py, Some(error));
                Err(replacement)
            } else {
                Err(error)
            }
        }
    }
}
#[pyfunction]
fn encoder_execute_native(
    kind: &str,
    module: &Bound<'_, PyAny>,
    values: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    Ok(match kind {
        "conditioning" => conditioning::conditioning_execute(module, values, false)?,
        "conditioning-strategy" => conditioning::conditioning_execute(module, values, true)?,
        "qwen" => qwen::qwen_execute(module, values)?,
        "minimax-reference" => minimax::minimax_execute(module, values, true)?,
        "minimax-i2v" => minimax::minimax_execute(module, values, false)?,
        "prompt" => prompt::prompt_execute(module, values)?,
        _ => return Err(PyValueError::new_err("Unknown remote execution kind")),
    }
    .unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encoder_run_native, m)?)?;
    m.add_function(wrap_pyfunction!(encoder_execute_native, m)?)?;
    prompt::register(m)?;
    qwen::register(m)?;
    Ok(())
}

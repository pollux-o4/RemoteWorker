use pyo3::{prelude::*, types::*};
#[pyfunction]
fn qwen_preparation_inputs_native(values: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(values.py());
    for (key, value) in values.iter() {
        let key = key.extract::<String>()?;
        result.set_item(key.strip_prefix("images.").unwrap_or(&key), value)?;
    }
    Ok(result.unbind())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qwen_preparation_inputs_native, m)?)?;
    Ok(())
}

//! Early CPU previews; reachability and cancellation never mutate GPU execution globals.
use pyo3::{
    prelude::*,
    types::{PyDict, PyInt, PyList, PyTuple},
};
use std::collections::HashSet;
#[pyfunction]
fn pipeline_preview_cached(
    job: &Bound<'_, PyAny>,
    context: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    if job.is_none()
        || context.is_none()
        || !context
            .getattr("prompt_id")?
            .eq(job.getattr("prompt_id")?)?
    {
        return Ok(None);
    }
    let py = job.py();
    let index = context
        .getattr("list_index")
        .unwrap_or_else(|_| 0_i64.into_pyobject(py).unwrap().into_any());
    let index = if index.is_truthy()? {
        index
    } else {
        0_i64.into_pyobject(py)?.into_any()
    };
    let ui = job
        .getattr("previews")?
        .call_method1("pop", ((context.getattr("node_id")?, index), py.None()))?;
    Ok(if ui.is_none() {
        None
    } else {
        Some(ui.unbind())
    })
}
// Python host adapter ABI mirrors the native preview publication callback.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature=(adapter,job,dynprompt,decoder_id,result,extra,outputs,list_index=0))]
fn pipeline_preview_publish(
    adapter: &Bound<'_, PyAny>,
    job: &Bound<'_, PyAny>,
    dynprompt: &Bound<'_, PyAny>,
    decoder_id: &str,
    result: &Bound<'_, PyAny>,
    extra: &Bound<'_, PyDict>,
    outputs: &Bound<'_, PyAny>,
    list_index: i64,
) -> PyResult<()> {
    let py = adapter.py();
    let Some(client) = extra.get_item("client_id")?.filter(|v| !v.is_none()) else {
        return Ok(());
    };
    let values = if result.is_instance_of::<PyDict>() {
        result.call_method1("get", ("result", PyTuple::empty(py)))?
    } else {
        result.clone()
    };
    if !(values.is_instance_of::<PyTuple>() || values.is_instance_of::<PyList>())
        || values.len()? == 0
    {
        return Ok(());
    }
    let images = values.get_item(0)?;
    if images
        .getattr("device")
        .and_then(|v| v.getattr("type"))
        .and_then(|v| v.extract::<String>())
        .unwrap_or_default()
        != "cpu"
        || images
            .getattr("ndim")
            .and_then(|v| v.extract::<i64>())
            .unwrap_or(0)
            != 4
    {
        return Ok(());
    }
    let mut reachable = HashSet::new();
    let mut todo: Vec<String> = outputs.extract()?;
    while let Some(key) = todo.pop() {
        if !reachable.insert(key.clone()) {
            continue;
        }
        let node = dynprompt.call_method1("get_node", (&key,))?;
        for (_, value) in node
            .call_method1("get", ("inputs", PyDict::new(py)))?
            .cast::<PyDict>()?
            .iter()
        {
            if let Ok(link) = value.cast::<PyList>() {
                if link.len() == 2 && link.get_item(1)?.is_instance_of::<PyInt>() {
                    if let Ok(key) = link.get_item(0)?.extract::<String>() {
                        todo.push(key);
                    }
                }
            }
        }
    }
    let preview_class = adapter.getattr("preview_class")?;
    for key in dynprompt.call_method0("all_node_ids")?.try_iter()? {
        let key = key?;
        let name: String = key.extract()?;
        if !reachable.contains(&name) {
            continue;
        }
        let node = dynprompt.call_method1("get_node", (&key,))?;
        if !adapter
            .getattr("classes")?
            .call_method1("get", (node.get_item("class_type")?,))?
            .is(&preview_class)
        {
            continue;
        }
        let link = node.get_item("inputs")?.call_method1("get", ("images",))?;
        if !link.eq(PyList::new(
            py,
            [
                decoder_id.into_pyobject(py)?.into_any(),
                0_i64.into_pyobject(py)?.into_any(),
            ],
        )?)? {
            continue;
        }
        if job.getattr("cancel")?.call_method0("is_set")?.is_truthy()? {
            return Ok(());
        }
        let attempt = (|| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("prompt", dynprompt.getattr("original_prompt")?)?;
            kwargs.set_item(
                "extra_pnginfo",
                extra
                    .get_item("extra_pnginfo")?
                    .unwrap_or_else(|| py.None().into_bound(py)),
            )?;
            let rendered = adapter
                .getattr("render")?
                .call((preview_class.call0()?, &images), Some(&kwargs))?;
            if job.getattr("cancel")?.call_method0("is_set")?.is_truthy()? {
                return Ok(());
            }
            let ui = rendered.get_item("ui")?;
            let previews = job.getattr("previews")?;
            previews.set_item((&key, list_index), &ui)?;
            let combined = PyDict::new(py);
            let images = PyList::empty(py);
            for (node, value) in previews.cast::<PyDict>()?.iter() {
                if node.get_item(0)?.eq(&key)? {
                    for image in value
                        .call_method1("get", ("images", PyList::empty(py)))?
                        .try_iter()?
                    {
                        images.append(image?)?;
                    }
                }
            }
            combined.set_item("images", images)?;
            let data = PyDict::new(py);
            data.set_item("node", &key)?;
            data.set_item(
                "display_node",
                dynprompt.call_method1("get_display_node_id", (&key,))?,
            )?;
            data.set_item("output", combined)?;
            data.set_item("prompt_id", job.getattr("prompt_id")?)?;
            adapter
                .getattr("send")?
                .call1(("executed", data, &client))?;
            let nodes = job.getattr("preview_nodes")?;
            if !nodes.contains(&key)? {
                nodes.call_method1("append", (&key,))?;
            }
            Ok::<(), PyErr>(())
        })();
        if let Err(error) = attempt {
            let kwargs = PyDict::new(py);
            kwargs.set_item(
                "exc_info",
                (error.get_type(py), error.value(py), error.traceback(py)),
            )?;
            py.import("logging")?
                .call_method1("getLogger", ("remoteworker.pipeline.previews",))?
                .call_method(
                    "error",
                    ("Early preview failed; deferring to native output node",),
                    Some(&kwargs),
                )?;
        }
    }
    Ok(())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pipeline_preview_cached, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline_preview_publish, m)?)?;
    Ok(())
}

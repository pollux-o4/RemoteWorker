//! Queue graph planning without speculative execution of arbitrary node types.
use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use std::collections::HashSet;

fn inputs<'py>(node: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let value = node.call_method1("get", ("inputs", PyDict::new(node.py())))?;
    Ok(value.cast_into::<PyDict>()?)
}
#[pyfunction]
fn literal_inputs(node: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyDict>>> {
    let values = inputs(node)?;
    for (_, v) in values.iter() {
        if v.is_instance_of::<PyList>() || v.is_instance_of::<PyDict>() {
            return Ok(None);
        }
    }
    if values
        .get_item("cache_policy")?
        .is_some_and(|v| v.extract::<String>().is_ok_and(|v| v == "always_refresh"))
    {
        return Ok(None);
    }
    Ok(Some(values.copy()?.unbind()))
}
#[pyfunction]
fn fingerprint(node: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = node.py();
    let values = inputs(node)?.copy()?;
    values.del_item("timeout_seconds").ok();
    let data = PyTuple::new(py, [node.get_item("class_type")?, values.into_any()])?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("sort_keys", true)?;
    let encoded = py
        .import("json")?
        .call_method("dumps", (data,), Some(&kwargs))?
        .call_method0("encode")?;
    py.import("hashlib")?
        .call_method1("sha256", (encoded,))?
        .call_method0("hexdigest")?
        .extract()
}
#[pyfunction]
fn is_sampling(graph: &Bound<'_, PyDict>, node_id: &Bound<'_, PyAny>) -> PyResult<bool> {
    let key = node_id.str()?;
    let Some(node) = graph.get_item(key)? else {
        return Ok(false);
    };
    let name: String = node.call_method1("get", ("class_type", ""))?.extract()?;
    Ok(name.contains("Sampler"))
}
#[pyfunction]
fn next_encoders(graph: &Bound<'_, PyDict>, encoders: HashSet<String>) -> PyResult<Py<PyList>> {
    let result = PyList::empty(graph.py());
    for (key, node) in graph.iter() {
        let name = node
            .call_method1("get", ("class_type",))?
            .extract::<String>()
            .ok();
        if name.is_some_and(|s| encoders.contains(&s)) {
            result.append((key, node.clone(), literal_inputs(&node)?))?;
        }
    }
    Ok(result.unbind())
}
#[pyfunction]
fn first_decoder(
    graph: &Bound<'_, PyDict>,
    decoders: std::collections::HashMap<String, String>,
) -> PyResult<Option<(String, String, String, String)>> {
    for image in [true, false] {
        for (key, node) in graph.iter() {
            let name = node
                .call_method1("get", ("class_type",))?
                .extract::<String>()
                .ok();
            let Some(kind) = name.as_ref().and_then(|n| decoders.get(n)) else {
                continue;
            };
            if (kind == "image") != image {
                continue;
            }
            let values = inputs(&node)?;
            if values.get_item("warmup")?.is_some_and(|v| !v.is_none()) {
                continue;
            }
            let Some(vae) = values
                .get_item("vae_name")?
                .and_then(|v| v.extract::<String>().ok())
                .filter(|s| !s.is_empty())
            else {
                continue;
            };
            let Some(url) = values
                .get_item("server_url")?
                .and_then(|v| v.extract::<String>().ok())
                .filter(|s| !s.is_empty())
            else {
                continue;
            };
            return Ok(Some((key.extract()?, url, vae, kind.clone())));
        }
    }
    Ok(None)
}
#[pyfunction]
#[pyo3(signature=(graph,outputs=None))]
fn reachable(graph: &Bound<'_, PyDict>, outputs: Option<Vec<Py<PyAny>>>) -> PyResult<Py<PyDict>> {
    let Some(mut todo) = outputs else {
        return Ok(graph.clone().unbind());
    };
    let py = graph.py();
    let mut visited = HashSet::<String>::new();
    while let Some(key) = todo.pop() {
        let key: String = key.bind(py).str()?.extract()?;
        if visited.contains(&key) {
            continue;
        }
        let Some(node) = graph.get_item(&key)? else {
            continue;
        };
        visited.insert(key);
        for (_, v) in inputs(&node)?.iter() {
            if let Ok(link) = v.cast::<PyList>() {
                if link.len() == 2 {
                    let upstream = link.get_item(0)?;
                    if graph.contains(upstream.str()?)? {
                        todo.push(upstream.unbind());
                    }
                }
            }
        }
    }
    let result = PyDict::new(py);
    for (key, node) in graph.iter() {
        if visited.contains(&key.extract::<String>()?) {
            result.set_item(key, node)?;
        }
    }
    Ok(result.unbind())
}
#[pyfunction]
fn graph_contract(graph: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(graph.py());
    for (key, node) in graph.iter() {
        let contract = PyDict::new(graph.py());
        contract.set_item("class_type", node.call_method1("get", ("class_type",))?)?;
        contract.set_item("inputs", inputs(&node)?)?;
        result.set_item(key, contract)?;
    }
    Ok(result.unbind())
}
#[pyfunction]
fn head(pending: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    let mut best: Option<Bound<'_, PyAny>> = None;
    for item in pending.try_iter()? {
        let item = item?;
        if best.as_ref().is_none_or(|b| {
            item.get_item(0)
                .and_then(|v| v.lt(b.get_item(0)?))
                .unwrap_or(false)
        }) {
            best = Some(item);
        }
    }
    Ok(best.map(Bound::unbind))
}
#[pyfunction]
fn job_graph(job: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let graph = job.get_item(2)?.cast_into::<PyDict>()?;
    let outputs = if job.len()? > 4 {
        job.get_item(4)?.extract()?
    } else {
        None
    };
    reachable(&graph, outputs)
}
#[pyfunction]
fn pending_encoders(
    pending: &Bound<'_, PyAny>,
    completed: &Bound<'_, PyAny>,
    encoders: HashSet<String>,
) -> PyResult<Py<PyList>> {
    let result = PyList::empty(pending.py());
    let Some(following) = head(pending)? else {
        return Ok(result.unbind());
    };
    let following = following.bind(pending.py());
    let graph = job_graph(following)?;
    for row in next_encoders(graph.bind(pending.py()), encoders)?
        .bind(pending.py())
        .iter()
    {
        let id = row.get_item(0)?;
        let node = row.get_item(1)?;
        let values = row.get_item(2)?;
        let token = PyTuple::new(
            pending.py(),
            [
                following.get_item(1)?,
                id.clone(),
                fingerprint(&node)?.into_pyobject(pending.py())?.into_any(),
            ],
        )?;
        if !completed.contains(&token)? {
            result.append((following, graph.bind(pending.py()), id, node, values, token))?;
        }
    }
    Ok(result.unbind())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(literal_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(is_sampling, m)?)?;
    m.add_function(wrap_pyfunction!(next_encoders, m)?)?;
    m.add_function(wrap_pyfunction!(first_decoder, m)?)?;
    m.add_function(wrap_pyfunction!(reachable, m)?)?;
    m.add_function(wrap_pyfunction!(graph_contract, m)?)?;
    m.add_function(wrap_pyfunction!(head, m)?)?;
    m.add_function(wrap_pyfunction!(job_graph, m)?)?;
    m.add_function(wrap_pyfunction!(pending_encoders, m)?)?;
    Ok(())
}

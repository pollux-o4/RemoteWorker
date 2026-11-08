use super::*;
use pyo3::{exceptions::*, prelude::*, types::*};
pub(super) fn conditioning_run<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let strategy = field(v, "conditioning_strategy")?.extract::<String>()?;
    let frames = if strategy == "text" {
        PyList::empty(m.py()).into_any()
    } else {
        let images = field(v, "images")?.cast_into::<PyDict>()?;
        m.getattr("ordered_image_frames")?
            .call((field(v, "image_count")?,), Some(&images))?
    };
    let clip = clip(m, v, false, true)?;
    let cache = field(v, "cache_policy")?.extract::<String>()? == "reuse_same_inputs";
    let args = selected(
        v,
        &[
            "prompt",
            "grounding_px",
            "system_prompt",
            "server_url",
            "peer_memory_policy",
        ],
    )?;
    args.set_item("strategy_name", &strategy)?;
    args.set_item("clip", clip)?;
    args.set_item("frames", frames)?;
    let key = if cache {
        let key = call(m, "_conditioning_cache_key", &args)?;
        let cached = m
            .getattr("_get_cached_conditioning_bytes")?
            .call1((&key,))?;
        if !cached.is_none() {
            return Ok(PyTuple::new(m.py(), [decode_conditioning(m.py(), &cached)?])?.into_any());
        }
        Some(key)
    } else {
        validate_policy(m, v, "cache_policy", "CACHE_POLICIES")?;
        None
    };
    args.del_item("server_url")?;
    args.set_item("api", api(m, v)?)?;
    args.set_item("request_id", uuid(m)?)?;
    let data = call(m, "execute_strategy_bytes", &args)?;
    if let Some(key) = key {
        m.getattr("_put_cached_conditioning_bytes")?
            .call1((key, &data))?;
    }
    Ok(PyTuple::new(m.py(), [decode_conditioning(m.py(), &data)?])?.into_any())
}

pub(super) fn conditioning_execute<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
    strategy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    validate_policy(m, v, "peer_memory_policy", "PEER_MEMORY_POLICIES")?;
    let api = field(v, "api")?;
    let name = if strategy {
        field(v, "strategy_name")?.extract::<String>()?
    } else {
        "text".into()
    };
    if name == "text" {
        if strategy && field(v, "frames")?.len()? > 0 {
            return Err(PyValueError::new_err(
                "Remote Conditioning text strategy requires 0 images",
            ));
        }
        release_preflight(m, &api, &field(v, "peer_memory_policy")?)?;
        let workflow = m.getattr("build_workflow")?.call1((
            field(v, "clip")?,
            field(v, "prompt")?,
            field(v, "request_id")?,
            field(v, "peer_memory_policy")?,
        ))?;
        return artifacts(
            m,
            v,
            &workflow,
            vec![m.getattr("CONDITIONING_ARTIFACT")?],
            |blobs| blobs.get_item("conditioning"),
        );
    }
    if name != "krea2-grounded" {
        return Err(PyValueError::new_err(format!(
            "Remote Conditioning strategy is not registered: {name:?}"
        )));
    }
    let frames = field(v, "frames")?;
    let count = frames.len()?;
    if !(1..=2).contains(&count) {
        return Err(PyValueError::new_err(format!(
            "Remote Conditioning {name} requires 1 to 2 images; got {count}"
        )));
    }
    let spec = m.getattr("STRATEGIES")?.get_item(&name)?;
    let capability = api.call_method1("require_node", (spec.getattr("remote_node")?,))?;
    m.getattr("validate_strategy_capability")?
        .call1((&spec, capability, count))?;
    release_preflight(m, &api, &field(v, "peer_memory_policy")?)?;
    let encoded = PyList::empty(m.py());
    for frame in frames.try_iter()? {
        encoded.append(m.getattr("encode_image_frame_png")?.call1((frame?,))?)?;
    }
    let request = m
        .getattr("RequestInputs")?
        .call1((&api, field(v, "request_id")?))?;
    with_context(&request, |request| {
        let refs = PyList::empty(m.py());
        for (i, png) in encoded.iter().enumerate() {
            let args = PyDict::new(m.py());
            args.set_item("index", i + 1)?;
            args.set_item("png", png)?;
            refs.append(request.call_method("upload_image", (), Some(&args))?)?;
        }
        let args = selected(
            v,
            &[
                "strategy_name",
                "clip",
                "prompt",
                "request_id",
                "grounding_px",
                "system_prompt",
                "peer_memory_policy",
            ],
        )?;
        args.set_item("image_refs", refs)?;
        let workflow = call(m, "build_strategy_workflow", &args)?;
        artifacts(
            m,
            v,
            &workflow,
            vec![m.getattr("CONDITIONING_ARTIFACT")?],
            |blobs| blobs.get_item("conditioning"),
        )
    })
}

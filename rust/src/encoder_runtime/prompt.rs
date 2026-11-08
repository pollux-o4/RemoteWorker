use super::*;
use pyo3::{prelude::*, types::*};
pub(super) fn prompt_execute<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let api = field(v, "api")?;
    let profile = api.call_method1("get_prompt_enhancer_profile", (field(v, "profile_id")?,))?;
    if !profile.is_instance(&m.getattr("PromptEnhancerProfile")?)? {
        return Err(PyRuntimeError::new_err(
            "Remote Prompt Enhancer profile response is invalid",
        ));
    }
    api.call_method1("require_node", ("RemotePromptGenerate",))?;
    api.call_method1("require_node", ("SaveRemoteText",))?;
    let images = field(v, "images")?;
    let videos = field(v, "videos")?;
    let images = if images.is_none() {
        PyList::empty(m.py()).into_any()
    } else {
        images
    };
    let videos = if videos.is_none() {
        PyList::empty(m.py()).into_any()
    } else {
        videos
    };
    crate::encoder_policy::validate_prompt_media_native(&profile, images.len()?, &videos)?;
    let request = m
        .getattr("RequestInputs")?
        .call1((&api, field(v, "request_id")?))?;
    with_context(&request, |request| {
        let refs = PyList::empty(m.py());
        for (i, image) in images.try_iter()?.enumerate() {
            refs.append(upload_image(m, request, i + 1, &image?)?)?;
        }
        let video_refs = PyList::empty(m.py());
        for (vi, video) in videos.try_iter()?.enumerate() {
            let video = video?;
            let paths = PyList::empty(m.py());
            let count = video.getattr("shape")?.get_item(0)?.extract::<usize>()?;
            for fi in 0..count {
                let data = m
                    .getattr("encode_image_frame_png")?
                    .call1((video.get_item(fi)?,))?;
                paths.append(upload_file(
                    request,
                    &format!("prompt-video-{vi:02}-frame-{fi:03}.png"),
                    &data,
                    "image/png",
                )?)?;
            }
            video_refs.append(paths)?;
        }
        let kwargs = PyDict::new(m.py());
        kwargs.set_item("image_refs", refs)?;
        kwargs.set_item("video_refs", video_refs)?;
        let workflow = m.getattr("build_workflow")?.call(
            (&profile, field(v, "prompt")?, field(v, "request_id")?),
            Some(&kwargs),
        )?;
        artifacts(
            m,
            v,
            &workflow,
            vec![m.getattr("PROMPT_ARTIFACT")?],
            |blobs| {
                m.getattr("decode_generated_prompt")?
                    .call1((blobs.get_item("prompt")?,))
            },
        )
    })
}
#[pyfunction]

pub(super) fn prompt_run<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let args = selected(v, &["profile_id", "prompt"])?;
    args.set_item("request_id", uuid(m)?)?;
    args.set_item(
        "api",
        m.getattr("PromptComfyApi")?
            .call1((field(v, "server_url")?, field(v, "timeout_seconds")?))?,
    )?;
    args.set_item(
        "images",
        m.getattr("_ordered_media")?
            .call1(("image", field(v, "media")?, 8))?,
    )?;
    args.set_item(
        "videos",
        m.getattr("_ordered_batches")?
            .call1(("video", field(v, "media")?, 3))?,
    )?;
    Ok(PyTuple::new(m.py(), [call(m, "execute", &args)?])?.into_any())
}
#[pyfunction]
#[allow(clippy::too_many_arguments)] // Comfy node ABI; field names preserved for saved workflows.
fn prompt_generate_native(
    module: &Bound<'_, PyAny>,
    clip: &Bound<'_, PyAny>,
    prompt: &Bound<'_, PyAny>,
    system_prompt: &Bound<'_, PyAny>,
    media_strategy: &str,
    max_length: &Bound<'_, PyAny>,
    image: Option<&Bound<'_, PyAny>>,
    media: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let py = module.py();
    let text = module
        .getattr("build_generation_prompt")?
        .call1((prompt, system_prompt))?;
    let max = max_length
        .extract::<u16>()
        .ok()
        .filter(|n| (16..=2048).contains(n));
    if max.is_none() || max_length.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyValueError::new_err(
            "Remote Prompt Generate max_length must be 16 to 2048",
        ));
    }
    let videos = PyList::empty(py);
    for index in 0..3 {
        if let Some(video) = media.get_item(format!("video_{index}"))? {
            if !video.is_none() {
                videos.append(video)?;
            }
        }
    }
    let kwargs = PyDict::new(py);
    match media_strategy {
        "qwen-image" => {
            if !videos.is_empty() {
                return Err(PyValueError::new_err(
                    "Remote Prompt Generate qwen-image does not accept video",
                ));
            }
            kwargs.set_item("image", image)?;
            kwargs.set_item("skip_template", false)?;
            kwargs.set_item("min_length", 1)?;
            kwargs.set_item("thinking", false)?;
        }
        "minimax-ref2va" => {
            let refs = PyList::empty(py);
            if let Some(image) = image.filter(|v| !v.is_none()) {
                let count = image.getattr("shape")?.get_item(0)?.extract::<usize>()?;
                for i in 0..count {
                    let item = PyDict::new(py);
                    item.set_item("type", "image")?;
                    item.set_item(
                        "data",
                        image.get_item(pyo3::types::PySlice::new(
                            py,
                            i as isize,
                            i as isize + 1,
                            1,
                        ))?,
                    )?;
                    refs.append(item)?;
                }
            }
            for video in videos.iter() {
                let item = PyDict::new(py);
                item.set_item("type", "video")?;
                item.set_item("data", video)?;
                refs.append(item)?;
            }
            kwargs.set_item("minimax_ref_items", refs)?;
        }
        _ => {
            return Err(PyValueError::new_err(
                "Remote Prompt Generate media strategy is invalid",
            ));
        }
    }
    let tokens = clip.call_method("tokenize", (text,), Some(&kwargs))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("do_sample", false)?;
    kwargs.set_item("max_length", max.unwrap())?;
    let generated = clip.call_method("generate", (tokens,), Some(&kwargs))?;
    Ok(
        PyTuple::new(py, [clip.call_method1("decode", (generated,))?])?
            .into_any()
            .unbind(),
    )
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prompt_generate_native, m)?)?;
    Ok(())
}

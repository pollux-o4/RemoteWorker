use super::*;
use pyo3::{exceptions::*, prelude::*, types::*};
#[pyclass]
struct DecodeCapture {
    module: Py<PyAny>,
    captured: Option<(Py<PyAny>, Py<PyAny>)>,
}
#[pymethods]
impl DecodeCapture {
    fn __call__(
        &mut self,
        py: Python<'_>,
        conditioning: &Bound<'_, PyAny>,
        latent: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let bytes = py.import("builtins")?.getattr("bytes")?;
        let a = bytes.call1((conditioning,))?.unbind();
        let b = bytes.call1((latent,))?.unbind();
        self.captured = Some((a, b));
        Ok(self
            .module
            .bind(py)
            .getattr("decode_remote_outputs")?
            .call1((conditioning, latent))?
            .unbind())
    }
}
pub(super) fn minimax_run<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
    reference: bool,
) -> PyResult<Bound<'py, PyAny>> {
    m.getattr("validate_cache_policy")?
        .call1((field(v, "cache_policy")?,))?;
    let media = if reference {
        field(v, "media")?.cast_into::<PyDict>()?.copy()?
    } else {
        PyDict::new(m.py())
    };
    let normalized = v.copy()?;
    for (name, override_name) in if reference {
        vec![
            ("vae_name", "vae_name_override"),
            ("audio_vae_name", "audio_vae_name_override"),
        ]
    } else {
        vec![("vae_name", "vae_name_override")]
    } {
        let override_value = if reference {
            let value = media
                .get_item(override_name)?
                .unwrap_or_else(|| m.py().None().into_bound(m.py()));
            media.del_item(override_name).ok();
            value
        } else {
            field(v, override_name)?
        };
        let kwargs = selected(v, &["unique_id", "extra_pnginfo"])?;
        kwargs.set_item(
            "input_name",
            if override_value.is_none() {
                name
            } else {
                override_name
            },
        )?;
        let value = m
            .getattr("resolve_vae_name")?
            .call((field(v, name)?, override_value), Some(&kwargs))?;
        normalized.set_item(name, value)?;
    }
    let clip = clip(m, v, false, false)?;
    let args = selected(
        &normalized,
        &[
            "vae_name",
            "prompt",
            "width",
            "height",
            "length",
            "peer_memory_policy",
        ],
    )?;
    args.set_item("clip", clip)?;
    if reference {
        args.set_item("audio_vae_name", field(&normalized, "audio_vae_name")?)?;
        args.set_item("ref_image_size", field(v, "ref_image_size")?)?;
        for (key, prefix, batches, max) in [
            ("images", "ref_image", false, 9),
            ("videos", "ref_video", true, 3),
        ] {
            let kwargs = PyDict::new(m.py());
            kwargs.set_item("batches", batches)?;
            kwargs.set_item("maximum", max)?;
            let items = m
                .getattr("_ordered_images")?
                .call((prefix, &media), Some(&kwargs))?;
            args.set_item(key, items)?;
        }
        for (key, prefix) in [("video_audios", "ref_video_audio"), ("audios", "ref_audio")] {
            args.set_item(
                key,
                m.getattr("_ordered_audio")?.call1((prefix, &media, 3))?,
            )?;
        }
        if field(&args, "video_audios")?.len()? > field(&args, "videos")?.len()? {
            return Err(PyValueError::new_err(
                "Remote MiniMax H3 video audio requires a same-index reference video",
            ));
        }
    } else {
        args.set_item("first_frame", field(v, "first_frame")?)?;
        args.set_item("last_frame", field(v, "last_frame")?)?;
    }
    let cache = field(v, "cache_policy")?.extract::<String>()? == "reuse_same_inputs";
    let key = if cache {
        args.set_item("server_url", field(v, "server_url")?)?;
        let key = call(
            m,
            if reference {
                "_reference_cache_key"
            } else {
                "_i2v_cache_key"
            },
            &args,
        )?;
        args.del_item("server_url")?;
        let cached = m.getattr("get_cached_minimax_outputs")?.call1((&key,))?;
        if !cached.is_none() {
            return m
                .getattr("decode_remote_outputs")?
                .call(cached.cast::<PyTuple>()?, None);
        }
        Some(key)
    } else {
        None
    };
    let capture = Py::new(
        m.py(),
        DecodeCapture {
            module: m.clone().unbind(),
            captured: None,
        },
    )?;
    args.set_item("deserialize", capture.bind(m.py()))?;
    args.set_item("request_id", uuid(m)?)?;
    args.set_item("api", api(m, v)?)?;
    let result = call(m, "execute", &args)?;
    if let Some(key) = key {
        if let Some((a, b)) = &capture.borrow(m.py()).captured {
            m.getattr("put_cached_minimax_outputs")?
                .call1((key, a, b))?;
        }
    }
    Ok(result)
}
#[pyfunction]

pub(super) fn minimax_execute<'py>(
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyDict>,
    reference: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let api = field(v, "api")?;
    let info = api.call_method1(
        "require_node",
        (if reference {
            "MiniMaxH3ReferenceToVideo"
        } else {
            "MiniMaxH3ImageToVideo"
        },),
    )?;
    if reference {
        m.getattr("validate_ref2va_capability")?.call1((info,))?;
    } else {
        let outputs = info.call_method1("get", ("output", PyList::empty(m.py())))?;
        if !outputs.contains("CONDITIONING")? || !outputs.contains("LATENT")? {
            return Err(PyRuntimeError::new_err(
                "Remote MiniMax H3 Image to Video peer is incompatible",
            ));
        }
    }
    m.getattr("validate_release_cleanup_capability")?
        .call1((&api, field(v, "peer_memory_policy")?))?;
    if reference {
        m.getattr("validate_peer_inputs")?.call1((
            &api,
            field(v, "clip")?,
            field(v, "vae_name")?,
            field(v, "audio_vae_name")?,
        ))?;
    } else {
        m.getattr("validate_peer_inputs")?.call1((
            &api,
            field(v, "clip")?,
            field(v, "vae_name")?,
        ))?;
    }
    let request = m
        .getattr("RequestInputs")?
        .call1((&api, field(v, "request_id")?))?;
    with_context(&request, |request| {
        let args = selected(
            v,
            &[
                "clip",
                "vae_name",
                "prompt",
                "width",
                "height",
                "length",
                "request_id",
                "peer_memory_policy",
            ],
        )?;
        if reference {
            args.set_item("audio_vae_name", field(v, "audio_vae_name")?)?;
            args.set_item("ref_image_size", field(v, "ref_image_size")?)?;
            let images = PyList::empty(m.py());
            for (i, image) in field(v, "images")?.try_iter()?.enumerate() {
                images.append(upload_image(m, request, i + 1, &image?.get_item(0)?)?)?;
            }
            args.set_item("image_refs", images)?;
            let videos = PyList::empty(m.py());
            for (vi, video) in field(v, "videos")?.try_iter()?.enumerate() {
                let video = video?;
                let paths = PyList::empty(m.py());
                let count = video.getattr("shape")?.get_item(0)?.extract::<usize>()?;
                for fi in 0..count {
                    let data = m
                        .getattr("encode_image_frame_png")?
                        .call1((video.get_item(fi)?,))?;
                    paths.append(upload_file(
                        request,
                        &format!("video-{vi:02}-frame-{fi:04}.png"),
                        &data,
                        "image/png",
                    )?)?;
                }
                videos.append(paths)?;
            }
            args.set_item("video_refs", videos)?;
            for (source, target, prefix) in [
                ("video_audios", "video_audio_refs", "video-audio"),
                ("audios", "audio_refs", "audio"),
            ] {
                let refs = PyList::empty(m.py());
                for (i, audio) in field(v, source)?.try_iter()?.enumerate() {
                    let data = m.getattr("encode_audio_wav")?.call1((audio?,))?;
                    refs.append(upload_file(
                        request,
                        &format!("{prefix}-{i:02}.wav"),
                        &data,
                        "audio/wav",
                    )?)?;
                }
                args.set_item(target, refs)?;
            }
        } else {
            for name in ["first_frame", "last_frame"] {
                let image = field(v, name)?;
                let path = if image.is_none() {
                    m.py().None().into_bound(m.py())
                } else {
                    let frame = m.getattr("_single_frame")?.call1((image, name))?;
                    let data = m.getattr("encode_image_frame_png")?.call1((frame,))?;
                    upload_file(request, &format!("{name}.png"), &data, "image/png")?
                };
                args.set_item(name, path)?;
            }
        }
        let workflow = call(m, "build_workflow", &args)?;
        artifacts(
            m,
            v,
            &workflow,
            vec![
                m.getattr("CONDITIONING_ARTIFACT")?,
                m.getattr("LATENT_ARTIFACT")?,
            ],
            |blobs| {
                field(v, "deserialize")?
                    .call1((blobs.get_item("conditioning")?, blobs.get_item("latent")?))
            },
        )
    })
}

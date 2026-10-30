//! Finalization transaction: validate source receipts, publish, then consume inputs.
use super::{secure_fs, store};
use crate::{from_python, to_python};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};
fn source(path: &Path, filename: &str) -> PyResult<String> {
    let id = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|s| s.to_str())
        .ok_or_else(|| PyValueError::new_err("공유 VAE가 반환한 파일 경로를 연결하세요."))?;
    let expected = store::path("outputs", id, filename)?;
    if path != expected {
        return Err(PyValueError::new_err(
            "공유 VAE가 반환한 파일 경로를 연결하세요.",
        ));
    }
    Ok(id.into())
}
fn duration(info: &Value) -> f64 {
    info.pointer("/format/duration")
        .and_then(|v| {
            v.as_str()
                .and_then(|s| s.parse().ok())
                .or_else(|| v.as_f64())
        })
        .unwrap_or(0.0)
}
fn validate_info(info: &Value, audio_required: bool) -> PyResult<Value> {
    let streams = info
        .get("streams")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let videos: Vec<_> = streams
        .iter()
        .filter(|s| s["codec_type"] == "video")
        .collect();
    let audios: Vec<_> = streams
        .iter()
        .filter(|s| s["codec_type"] == "audio")
        .collect();
    let duration = duration(info);
    if videos.len() != 1 || !duration.is_finite() || duration <= 0.0 {
        return Err(PyValueError::new_err(
            "완성 영상의 비디오 트랙·길이를 확인할 수 없습니다.",
        ));
    }
    if audio_required
        && (audios.is_empty()
            || audios.iter().any(|s| {
                s["sample_rate"]
                    .as_str()
                    .and_then(|s| s.parse::<i64>().ok())
                    .or_else(|| s["sample_rate"].as_i64())
                    .unwrap_or(0)
                    <= 0
            }))
    {
        return Err(PyValueError::new_err(
            "완성 영상에 오디오 트랙이 없습니다. 중간 파일을 보존합니다.",
        ));
    }
    Ok(
        json!({"audio":!audios.is_empty(),"duration":duration,"width":videos[0].get("width"),"height":videos[0].get("height")}),
    )
}
#[pyfunction]
fn vae_mux_source_id(path: &Bound<'_, PyAny>, filename: &str) -> PyResult<String> {
    source(&PathBuf::from(path.str()?.to_str()?), filename)
}
#[pyfunction]
fn vae_mux_command(
    py: Python<'_>,
    video: &Bound<'_, PyAny>,
    audio: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
) -> PyResult<Py<PyList>> {
    let video: String = video.str()?.extract()?;
    let target: String = target.str()?.extract()?;
    let mut command: Vec<String> = ["ffmpeg", "-v", "error", "-nostdin", "-i"]
        .map(str::to_owned)
        .to_vec();
    command.push(video);
    if audio.is_truthy()? {
        command.push("-i".into());
        command.push(audio.str()?.extract()?);
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-af",
                "apad",
                "-shortest",
            ]
            .map(str::to_owned),
        );
    } else {
        command.extend(["-map", "0:v:0", "-map", "0:a?", "-c", "copy"].map(str::to_owned));
    }
    command.extend(["-movflags", "+faststart", "-f", "mp4", "-y"].map(str::to_owned));
    command.push(target);
    Ok(PyList::new(py, command)?.unbind())
}
#[pyfunction]
fn vae_mux_validate(info: &Bound<'_, PyAny>, audio_required: bool) -> PyResult<Py<PyAny>> {
    to_python(
        info.py(),
        &validate_info(&from_python(info)?, audio_required)?,
    )
}
#[pyfunction]
fn vae_mux_combine(
    py: Python<'_>,
    video_path: &str,
    audio_path: &str,
    cleanup: bool,
    interrupt: &Bound<'_, PyAny>,
    encode: &Bound<'_, PyAny>,
    probe: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let video = Path::new(video_path);
    let video_id = source(video, "video.mp4")?;
    let audio = (!audio_path.is_empty()).then(|| Path::new(audio_path));
    let audio_id = audio.map(|p| source(p, "audio.wav")).transpose()?;
    let mut sources = vec![video_id];
    if let Some(id) = audio_id {
        sources.push(id);
    }
    let key = format!("mux-v1:{}", sources.join(":"));
    let digest: String = py
        .import("hashlib")?
        .call_method1("sha256", (pyo3::types::PyBytes::new(py, key.as_bytes()),))?
        .call_method0("hexdigest")?
        .extract()?;
    let identity = &digest[..32];
    let options = json!({"kind":"image","sources":sources,"mux":"h264-copy-aac-v1"});
    let ordered: BTreeSet<_> = sources
        .iter()
        .map(String::as_str)
        .chain([identity])
        .collect();
    let _locks = ordered
        .into_iter()
        .map(|id| store::acquire(py, id))
        .collect::<PyResult<Vec<_>>>()?;
    let receipt = if let Some(receipt) = store::read(identity, Some(&options))? {
        receipt.document
    } else {
        let mut committed_sources = Vec::new();
        for (id, filename) in sources.iter().zip(["video.mp4", "audio.wav"]) {
            let expected = store::path("outputs", id, filename)?;
            let committed = store::read(id, None)?
                .filter(|receipt| receipt.path == expected && receipt.media_file.is_some())
                .ok_or_else(|| {
                    PyValueError::new_err("디코딩 결과가 없습니다. 원격 VAE를 다시 실행하세요.")
                })?;
            committed_sources.push(committed);
        }
        let pinned_video =
            secure_fs::descriptor_path(committed_sources[0].media_file.as_ref().unwrap());
        let video = pinned_video.as_path();
        let pinned_audio = committed_sources
            .get(1)
            .map(|receipt| secure_fs::descriptor_path(receipt.media_file.as_ref().unwrap()));
        let audio = pinned_audio.as_deref();
        let video_info = from_python(&probe.call1((store::as_path(py, video)?, interrupt))?)?;
        let video_duration = duration(&video_info);
        if let Some(audio) = audio {
            let audio_info = from_python(&probe.call1((store::as_path(py, audio)?, interrupt))?)?;
            if !audio_info
                .get("streams")
                .and_then(Value::as_array)
                .is_some_and(|s| s.iter().any(|s| s["codec_type"] == "audio"))
            {
                return Err(PyValueError::new_err(
                    "연결된 음성 파일에 오디오 트랙이 없습니다.",
                ));
            }
            let audio_duration = duration(&audio_info);
            if !audio_duration.is_finite()
                || !video_duration.is_finite()
                || (video_duration - audio_duration).abs() > 0.25
            {
                return Err(PyValueError::new_err(
                    "영상과 음성 길이가 다릅니다. 중간 파일은 보존했습니다. 연결한 작업을 확인하세요.",
                ));
            }
        }
        let destination = store::media(identity, &options)?;
        let mut metadata = Value::Null;
        store::atomic(py, &destination, |temporary| {
            let audio = audio.map(|p| store::as_path(py, p)).transpose()?;
            encode.call1((
                store::as_path(py, video)?,
                audio,
                store::as_path(py, temporary)?,
                interrupt,
            ))?;
            let info = from_python(&probe.call1((store::as_path(py, temporary)?, interrupt))?)?;
            metadata = validate_info(&info, !audio_path.is_empty())?;
            if (metadata["duration"].as_f64().unwrap() - video_duration).abs() > 0.25 {
                return Err(PyValueError::new_err(
                    "결합 후 영상 길이가 달라졌습니다. 중간 파일을 보존합니다.",
                ));
            }
            Ok(())
        })?;
        let mut receipt = metadata.as_object().unwrap().clone();
        receipt.insert("request_id".into(), identity.into());
        receipt.insert("path".into(), destination.to_string_lossy().as_ref().into());
        receipt.insert(
            "bytes".into(),
            secure_fs::open(&destination)?.metadata()?.len().into(),
        );
        receipt.insert("options".into(), options.clone());
        receipt.insert("sources".into(), json!(sources));
        receipt.insert("final".into(), true.into());
        let receipt = Value::Object(receipt);
        store::write_json(py, &destination.with_file_name("receipt.json"), &receipt)?;
        receipt
    };
    if cleanup {
        for (id, filename) in sources.iter().zip(["video.mp4", "audio.wav"]) {
            let source = store::path("outputs", id, filename)?;
            store::write_json(
                py,
                &source.with_file_name("consumed.json"),
                &json!({"final_id":identity}),
            )?;
            store::remove_missing(&source)?;
            store::remove_missing(&store::path("temp", id, "latent.pt")?)?;
        }
    }
    to_python(py, &receipt)
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vae_mux_source_id, m)?)?;
    m.add_function(wrap_pyfunction!(vae_mux_command, m)?)?;
    m.add_function(wrap_pyfunction!(vae_mux_validate, m)?)?;
    m.add_function(wrap_pyfunction!(vae_mux_combine, m)?)?;
    Ok(())
}

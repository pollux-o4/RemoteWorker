use super::invalid;
use pyo3::{exceptions::*, prelude::*, types::*};
#[pyfunction]
fn encode_audio_wav_native(
    sample_rate: &Bound<'_, PyAny>,
    shape: Vec<usize>,
    data: Vec<u8>,
) -> PyResult<Vec<u8>> {
    let rate = sample_rate
        .extract::<u32>()
        .ok()
        .filter(|n| (1..=384000).contains(n))
        .ok_or_else(|| invalid("Remote MiniMax H3 audio sample_rate is invalid"))?;
    if shape.len() != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] == 0 {
        return Err(invalid(
            "Remote MiniMax H3 AUDIO waveform must have shape [1, channels, samples]",
        ));
    }
    let channels = shape[1];
    let samples = shape[2];
    let expected = channels
        .checked_mul(samples)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("Remote MiniMax H3 AUDIO waveform is too large"))?;
    if data.len() != expected {
        return Err(invalid(
            "Remote MiniMax H3 AUDIO waveform data is malformed",
        ));
    }
    let mut values = Vec::with_capacity(channels * samples);
    for bytes in data.chunks_exact(4) {
        let value = f32::from_le_bytes(bytes.try_into().unwrap());
        if !value.is_finite() {
            return Err(invalid(
                "Remote MiniMax H3 AUDIO waveform contains non-finite values",
            ));
        }
        values.push((value.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16);
    }
    let size = channels
        .checked_mul(samples)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n <= u32::MAX - 36)
        .ok_or_else(|| invalid("Remote MiniMax H3 AUDIO waveform is too large"))?;
    let channel_count = u16::try_from(channels)
        .ok()
        .filter(|c| *c <= u16::MAX / 2)
        .ok_or_else(|| invalid("Remote MiniMax H3 AUDIO channel count is invalid"))?;
    let align = channel_count * 2;
    let byte_rate = rate
        .checked_mul(align as u32)
        .ok_or_else(|| invalid("Remote MiniMax H3 AUDIO channel count is invalid"))?;
    let mut wav = Vec::with_capacity(44 + size as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(size + 36).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channel_count.to_le_bytes());
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&size.to_le_bytes());
    for i in 0..samples {
        for c in 0..channels {
            wav.extend_from_slice(&values[c * samples + i].to_le_bytes());
        }
    }
    Ok(wav)
}
#[pyfunction]
fn validate_latent_artifact_native(state: &Bound<'_, PyAny>, prefix: &str) -> PyResult<()> {
    if !state.is_instance_of::<PyDict>() || !state.contains("samples")? {
        return Err(PyRuntimeError::new_err(format!(
            "{prefix} latent artifact is invalid"
        )));
    }
    Ok(())
}
#[pyfunction]
fn peer_loaders_native(
    kind: &str,
    module: &Bound<'_, PyAny>,
    api: &Bound<'_, PyAny>,
    clip: &Bound<'_, PyAny>,
    vae_names: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let py = module.py();
    let qwen = kind == "qwen";
    let prefix = if qwen {
        "Remote Qwen Image 2.1"
    } else {
        "Remote MiniMax H3"
    };
    let options = api.call_method0("clip_loader_options")?;
    let clip = if qwen {
        let choices = options.call_method1("get", ("clip_name", PyList::empty(py)))?;
        let resolved = module.getattr("resolve_peer_model_name")?.call1((
            clip.getattr("clip_name")?,
            choices,
            "clip_name",
        ))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("clip_name", resolved)?;
        clip.call_method("_replace", (), Some(&kwargs))?
    } else {
        clip.clone()
    };
    for (i, name) in ["clip_name", "clip_type", "device"].iter().enumerate() {
        let value = clip.get_item(i)?;
        if !options
            .call_method1("get", (*name, PyList::empty(py)))?
            .contains(&value)?
        {
            return Err(invalid(format!(
                "{prefix} peer does not provide {name}: {}",
                value.repr()?.to_str()?
            )));
        }
    }
    let info = api.call_method1("require_node", ("VAELoader",))?;
    let parsed = (|| -> PyResult<Bound<'_, PyAny>> {
        info.get_item("input")?
            .get_item("required")?
            .get_item("vae_name")?
            .get_item(0)
    })();
    let options = parsed.map_err(|_| {
        PyRuntimeError::new_err(format!("{prefix} peer VAELoader capability is malformed"))
    })?;
    if qwen {
        if !options.is_instance_of::<PyList>() {
            return Err(PyRuntimeError::new_err(
                "Remote Qwen Image 2.1 peer VAELoader capability is malformed",
            ));
        }
        let vae = module.getattr("resolve_peer_model_name")?.call1((
            vae_names
                .first()
                .ok_or_else(|| invalid("Missing Qwen VAE name"))?,
            options,
            "vae_name",
        ))?;
        return Ok(PyTuple::new(py, [clip, vae])?.into_any().unbind());
    }
    if !options.is_instance_of::<PyList>() {
        return Err(invalid(
            "Remote MiniMax H3 peer does not provide the selected VAE",
        ));
    }
    for name in vae_names {
        if !options.contains(name.bind(py))? {
            return Err(invalid(
                "Remote MiniMax H3 peer does not provide the selected VAE",
            ));
        }
    }
    Ok(py.None())
}
#[pyfunction]
fn minimax_cached_native(cache: &Bound<'_, PyAny>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let value = cache.call_method1("get", (key,))?;
    if value.is_none() {
        return Ok(value.unbind());
    }
    if value.len()? != 2 {
        cache.call_method0("clear")?;
        return Ok(cache.py().None());
    }
    Ok(value.unbind())
}
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_audio_wav_native, m)?)?;
    m.add_function(wrap_pyfunction!(validate_latent_artifact_native, m)?)?;
    m.add_function(wrap_pyfunction!(peer_loaders_native, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_cached_native, m)?)?;
    Ok(())
}

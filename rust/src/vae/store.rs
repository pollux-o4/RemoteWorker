use super::{codec, secure_fs};
use crate::{from_python, to_python};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    os::fd::AsRawFd,
    path::{Component, Path, PathBuf},
};

/// Receipt whose request identity, owned path, size and options passed validation.
/// Raw peer JSON cannot enter acknowledge/consumption without this construction.
pub(super) struct CommittedReceipt {
    pub(super) document: Value,
    pub(super) path: PathBuf,
    pub(super) media_file: Option<File>,
}

#[derive(Clone, Debug)]
struct SharedRequest(String);
impl SharedRequest {
    fn parse(value: &str) -> PyResult<Self> {
        if value.len() != 32
            || !value
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(PyValueError::new_err("Invalid shared VAE request ID"));
        }
        Ok(Self(value.into()))
    }
}
fn root() -> PyResult<PathBuf> {
    let location = std::env::var("REMOTEWORKER_SHARED_VAE_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::var("CLUSTER_ROOT")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| {
                    Path::new(&s)
                        .join("shared/remote-vae")
                        .to_string_lossy()
                        .into_owned()
                })
        });
    let path = location
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .ok_or_else(|| {
            PyValueError::new_err(
                "REMOTEWORKER_SHARED_VAE_DIR에 두 서버가 공유하는 절대 경로를 설정하세요.",
            )
        })?;
    fs::create_dir_all(&path)?;
    Ok(fs::canonicalize(path)?)
}
fn safe_segment(value: &str) -> bool {
    !value.is_empty()
        && Path::new(value).components().count() == 1
        && matches!(
            Path::new(value).components().next(),
            Some(Component::Normal(_))
        )
}
pub(super) fn path(area: &str, id: &str, name: &str) -> PyResult<PathBuf> {
    let request = SharedRequest::parse(id)?;
    if !matches!(area, "temp" | "outputs") || !safe_segment(name) {
        return Err(PyValueError::new_err(
            "Shared VAE path must not contain symlinks",
        ));
    }
    let base = root()?;
    let mut path = base;
    for part in [area, request.0.as_str(), name] {
        path.push(part);
        if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(PyValueError::new_err(
                "Shared VAE path must not contain symlinks",
            ));
        }
    }
    Ok(path)
}
pub(super) fn as_path(py: Python<'_>, path: &Path) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("pathlib")?
        .getattr("Path")?
        .call1((path.to_string_lossy().as_ref(),))?
        .unbind())
}
fn input_path(value: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    Ok(PathBuf::from(value.str()?.to_str()?))
}
fn tensor(options: &Value) -> bool {
    options.get("delivery").and_then(Value::as_str) == Some("tensor")
}
pub(super) fn media(id: &str, options: &Value) -> PyResult<PathBuf> {
    let name = if tensor(options) {
        "decoded.pt"
    } else if options.get("kind").and_then(Value::as_str) == Some("audio") {
        "audio.wav"
    } else {
        "video.mp4"
    };
    path(if tensor(options) { "temp" } else { "outputs" }, id, name)
}
fn receipt(id: &str, options: Option<&Value>) -> PyResult<PathBuf> {
    let tensor = options.is_some_and(tensor);
    path(
        if tensor { "temp" } else { "outputs" },
        id,
        if tensor {
            "decoded-receipt.json"
        } else {
            "receipt.json"
        },
    )
}
fn json_read(path: &Path) -> PyResult<Value> {
    let bytes = secure_fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| PyValueError::new_err(e.to_string()))
}
pub(super) fn read(id: &str, options: Option<&Value>) -> PyResult<Option<CommittedReceipt>> {
    let receipt_path = receipt(id, options)?;
    if !receipt_path.exists() {
        return Ok(None);
    }
    let value = json_read(&receipt_path)?;
    let video = media(id, value.get("options").unwrap_or(&json!({})))?;
    let marker = receipt_path.with_file_name("consumed.json");
    let mut consumed = false;
    if !video.exists() && marker.is_file() {
        let marker = json_read(&marker)?;
        let final_id = marker
            .get("final_id")
            .and_then(Value::as_str)
            .ok_or_else(|| PyValueError::new_err("Invalid shared VAE request ID"))?;
        let final_path = receipt(final_id, None)?;
        if !final_path.with_file_name("consumed.json").exists() {
            if let Some(final_value) = read(final_id, None)? {
                consumed = final_value.document.get("final") == Some(&Value::Bool(true))
                    && final_value
                        .document
                        .get("sources")
                        .and_then(Value::as_array)
                        .is_some_and(|a| a.iter().any(|v| v.as_str() == Some(id)));
            }
        }
    }
    let media_file = match secure_fs::open(&video) {
        Ok(file) => Some(file),
        Err(error)
            if Python::attach(|py| {
                error.is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py)
            }) =>
        {
            None
        }
        Err(error) => return Err(error),
    };
    let valid_size = media_file
        .as_ref()
        .and_then(|file| file.metadata().ok())
        .is_some_and(|metadata| {
            metadata.len() > 0 && value.get("bytes").and_then(Value::as_u64) == Some(metadata.len())
        });
    if value.get("request_id").and_then(Value::as_str) != Some(id)
        || value.get("path").and_then(Value::as_str) != Some(video.to_string_lossy().as_ref())
        || (!consumed && !valid_size)
    {
        return Err(PyValueError::new_err(
            "공유 결과 경로 또는 파일 크기를 확인할 수 없습니다. latent를 보존합니다.",
        ));
    }
    if options.is_some_and(|options| value.get("options") != Some(options)) {
        return Err(PyValueError::new_err(
            "이미 완료된 작업의 설정이 다릅니다. 새 latent 저장 작업을 만드세요.",
        ));
    }
    Ok(Some(CommittedReceipt {
        document: value,
        path: video,
        media_file,
    }))
}
fn unique(py: Python<'_>) -> PyResult<String> {
    py.import("uuid")?
        .call_method0("uuid4")?
        .getattr("hex")?
        .extract()
}
pub(super) fn remove_missing(path: &Path) -> PyResult<()> {
    secure_fs::remove(path, true)
}
pub(super) fn atomic<F>(py: Python<'_>, path: &Path, writer: F) -> PyResult<()>
where
    F: FnOnce(&Path) -> PyResult<()>,
{
    let directory = secure_fs::parent(path, true)?;
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let temporary = std::ffi::OsString::from(format!(".{stem}-{}", unique(py)?));
    let file = directory.open(&temporary, libc::O_RDWR | libc::O_CREAT | libc::O_EXCL)?;
    let result = (|| {
        writer(&secure_fs::descriptor_path(&file))?;
        file.sync_all()?;
        directory.publish(
            &temporary,
            path.file_name()
                .ok_or_else(|| PyValueError::new_err("Invalid shared VAE destination"))?,
            &file,
        )
    })();
    let cleanup = directory.unlink(&temporary, true);
    match result {
        Ok(()) => cleanup,
        Err(error) => Err(error),
    }
}
pub(super) fn write_json(py: Python<'_>, path: &Path, value: &Value) -> PyResult<()> {
    atomic(py, path, |temporary| {
        fs::write(
            temporary,
            serde_json::to_vec(value).map_err(|e| PyValueError::new_err(e.to_string()))?,
        )?;
        Ok(())
    })
}

#[pyclass]
pub(super) struct VaeRequestLock {
    file: Option<File>,
}
#[pymethods]
impl VaeRequestLock {
    fn __enter__(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(file) = &self.file {
            py.import("fcntl")?
                .call_method1("flock", (file.as_raw_fd(), 2))?;
        }
        Ok(())
    }
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _trace: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let Some(file) = self.file.take() {
            py.import("fcntl")?
                .call_method1("flock", (file.as_raw_fd(), 8))?;
        }
        Ok(false)
    }
}
pub(super) fn acquire(py: Python<'_>, id: &str) -> PyResult<VaeRequestLock> {
    let file = path("temp", id, "request.lock")?;
    let file = secure_fs::create_lock(&file)?;
    py.import("fcntl")?
        .call_method1("flock", (file.as_raw_fd(), 2))?;
    Ok(VaeRequestLock { file: Some(file) })
}
#[pyfunction]
fn vae_shared_root(py: Python<'_>) -> PyResult<Py<PyAny>> {
    as_path(py, &root()?)
}
#[pyfunction]
fn vae_request_path(
    py: Python<'_>,
    area: &str,
    request_id: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Py<PyAny>> {
    let id = request_id
        .extract::<String>()
        .map_err(|_| PyValueError::new_err("Invalid shared VAE request ID"))?;
    as_path(py, &path(area, &id, name)?)
}
#[pyfunction]
fn vae_request_lock(_py: Python<'_>, request_id: &str) -> PyResult<VaeRequestLock> {
    let file = path("temp", request_id, "request.lock")?;
    Ok(VaeRequestLock {
        file: Some(secure_fs::create_lock(&file)?),
    })
}
#[pyfunction]
fn vae_atomic_write(destination: &Bound<'_, PyAny>, writer: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = destination.py();
    atomic(py, &input_path(destination)?, |path| {
        writer.call1((as_path(py, path)?,))?;
        Ok(())
    })
}
#[pyfunction]
fn vae_write_json(destination: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    write_json(
        destination.py(),
        &input_path(destination)?,
        &from_python(value)?,
    )
}
#[pyfunction]
fn vae_media_path(
    py: Python<'_>,
    request_id: &str,
    options: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    as_path(py, &media(request_id, &from_python(options)?)?)
}
#[pyfunction]
#[pyo3(signature=(request_id,options=None))]
fn vae_receipt_file(
    py: Python<'_>,
    request_id: &str,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let options = options.map(from_python).transpose()?;
    as_path(py, &receipt(request_id, options.as_ref())?)
}
#[pyfunction]
#[pyo3(signature=(request_id,options=None))]
fn vae_read_receipt(
    py: Python<'_>,
    request_id: &str,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Py<PyAny>>> {
    let options = options.map(from_python).transpose()?;
    read(request_id, options.as_ref())?
        .as_ref()
        .map(|v| to_python(py, &v.document))
        .transpose()
}
#[pyfunction]
pub(super) fn vae_acknowledge(
    py: Python<'_>,
    request_id: &str,
    reported: &Bound<'_, PyAny>,
    options: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let _lock = acquire(py, request_id)?;
    let options = from_python(options)?;
    let current = read(request_id, Some(&options))?;
    let reported = from_python(reported)?;
    let Some(current) = current.filter(|c| c.document == reported) else {
        return Err(PyValueError::new_err(
            "처리 서버와 호출 서버의 결과가 일치하지 않습니다. 공유 폴더를 확인하세요. latent를 보존합니다.",
        ));
    };
    remove_missing(&path("temp", request_id, "latent.pt")?)?;
    Ok(current.path.to_string_lossy().into_owned())
}
#[pyfunction]
#[pyo3(signature=(samples,audio=None))]
pub(super) fn vae_save_latent(
    samples: &Bound<'_, PyAny>,
    audio: Option<&Bound<'_, PyAny>>,
) -> PyResult<String> {
    let py = samples.py();
    codec::latent(samples)?;
    let value = samples.cast::<PyDict>()?.copy()?;
    if let Some(audio) = audio {
        value.set_item("remote_vae_audio", audio)?;
    }
    let value = codec::cpu(value.as_any())?;
    let id = unique(py)?;
    atomic(py, &path("temp", &id, "latent.pt")?, |p| {
        py.import("torch")?
            .call_method1("save", (value, as_path(py, p)?))?;
        Ok(())
    })?;
    Ok(id)
}
fn load(py: Python<'_>, id: &str) -> PyResult<Py<PyAny>> {
    let p = path("temp", id, "latent.pt")?;
    if !p.is_file() {
        return Err(PyValueError::new_err(format!(
            "공유 latent가 없습니다: {id}. 완료 후 삭제되었거나 저장에 실패했습니다."
        )));
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("map_location", "cpu")?;
    kwargs.set_item("weights_only", true)?;
    let value = py.import("torch")?.call_method(
        "load",
        (as_path(
            py,
            &secure_fs::descriptor_path(&secure_fs::open(&p)?),
        )?,),
        Some(&kwargs),
    )?;
    let value = codec::restore(&value)?;
    codec::latent(value.bind(py))?;
    Ok(value)
}
#[pyfunction]
fn vae_load_latent(py: Python<'_>, request_id: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let id = request_id
        .extract::<String>()
        .map_err(|_| PyValueError::new_err("Invalid shared VAE request ID"))?;
    load(py, &id)
}
#[pyfunction]
fn vae_publish(
    py: Python<'_>,
    request_id: &str,
    options: &Bound<'_, PyAny>,
    encode: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let _lock = acquire(py, request_id)?;
    let settings = from_python(options)?;
    if let Some(value) = read(request_id, Some(&settings))? {
        return to_python(py, &value.document);
    }
    let samples = load(py, request_id)?;
    let destination = media(request_id, &settings)?;
    let mut metadata = Value::Null;
    atomic(py, &destination, |p| {
        let result = encode.call1((samples, as_path(py, p)?, options))?;
        metadata = from_python(&result)?;
        Ok(())
    })?;
    let mut value = metadata
        .as_object()
        .cloned()
        .ok_or_else(|| PyValueError::new_err("Shared result writer returned invalid metadata"))?;
    let size = secure_fs::open(&destination)?.metadata()?.len();
    if size == 0 {
        return Err(PyValueError::new_err(
            "Shared result writer produced an empty file",
        ));
    }
    value.insert("request_id".into(), request_id.into());
    value.insert("path".into(), destination.to_string_lossy().as_ref().into());
    value.insert("bytes".into(), size.into());
    value.insert("options".into(), settings.clone());
    let value = Value::Object(value);
    write_json(py, &receipt(request_id, Some(&settings))?, &value)?;
    to_python(py, &value)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VaeRequestLock>()?;
    m.add_function(wrap_pyfunction!(vae_shared_root, m)?)?;
    m.add_function(wrap_pyfunction!(vae_request_path, m)?)?;
    m.add_function(wrap_pyfunction!(vae_request_lock, m)?)?;
    m.add_function(wrap_pyfunction!(vae_atomic_write, m)?)?;
    m.add_function(wrap_pyfunction!(vae_write_json, m)?)?;
    m.add_function(wrap_pyfunction!(vae_media_path, m)?)?;
    m.add_function(wrap_pyfunction!(vae_receipt_file, m)?)?;
    m.add_function(wrap_pyfunction!(vae_read_receipt, m)?)?;
    m.add_function(wrap_pyfunction!(vae_acknowledge, m)?)?;
    m.add_function(wrap_pyfunction!(vae_save_latent, m)?)?;
    m.add_function(wrap_pyfunction!(vae_load_latent, m)?)?;
    m.add_function(wrap_pyfunction!(vae_publish, m)?)?;
    Ok(())
}

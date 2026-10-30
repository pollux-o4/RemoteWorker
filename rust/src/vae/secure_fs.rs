//! Linux file effects pinned to directory/file descriptors, never re-resolved by a writer.
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{
    ffi::CString,
    fs::{File, OpenOptions},
    io::{self, Read},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt},
        },
    },
    path::{Component, Path, PathBuf},
};

pub(super) struct Directory {
    file: File,
    logical: PathBuf,
}
fn component(value: &std::ffi::OsStr) -> PyResult<CString> {
    CString::new(value.as_bytes())
        .map_err(|_| PyValueError::new_err("Shared VAE path contains a null byte"))
}
fn unsafe_path() -> PyErr {
    PyValueError::new_err("Shared VAE path changed or contains symlinks")
}
fn owned(fd: libc::c_int) -> io::Result<File> {
    if fd < 0 {
        return Err(io::Error::last_os_error());
    } // SAFETY: successful open/openat returns a new, uniquely owned descriptor.
    Ok(unsafe { File::from_raw_fd(fd) })
}
fn open_directory(path: &Path, create: bool) -> PyResult<Directory> {
    if !path.is_absolute() {
        return Err(unsafe_path());
    }
    let mut dir = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open("/")?;
    for part in path.components() {
        let Component::Normal(name) = part else {
            if matches!(part, Component::RootDir) {
                continue;
            }
            return Err(unsafe_path());
        };
        let name = component(name)?;
        if create {
            // SAFETY: dir is live and name is a valid, NUL-terminated single component.
            let created = unsafe { libc::mkdirat(dir.as_raw_fd(), name.as_ptr(), 0o700) };
            if created == 0 {
                dir.sync_all()?;
            } else {
                let error = io::Error::last_os_error();
                if error.kind() != io::ErrorKind::AlreadyExists {
                    return Err(error.into());
                }
            }
        }
        // SAFETY: directory descriptor and CString remain live during openat; no symlink traversal.
        dir = owned(unsafe {
            libc::openat(
                dir.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        })?;
    }
    Ok(Directory {
        file: dir,
        logical: path.to_owned(),
    })
}
pub(super) fn parent(path: &Path, create: bool) -> PyResult<Directory> {
    open_directory(path.parent().ok_or_else(unsafe_path)?, create)
}
impl Directory {
    pub(super) fn open(&self, name: &std::ffi::OsStr, flags: libc::c_int) -> PyResult<File> {
        let name = component(name)?;
        // SAFETY: descriptor/CString remain live; mode is used only for O_CREAT.
        let file = owned(unsafe {
            libc::openat(
                self.file.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
                0o600,
            )
        })?;
        if !file.metadata()?.is_file() {
            return Err(unsafe_path());
        }
        Ok(file)
    }
    fn same_file(&self, name: &std::ffi::OsStr, file: &File) -> PyResult<bool> {
        let name = component(name)?;
        let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: fstatat initializes stat on success; name/dir remain live.
        if unsafe {
            libc::fstatat(
                self.file.as_raw_fd(),
                name.as_ptr(),
                metadata.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        } < 0
        {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: successful fstatat above initialized every field.
        let metadata = unsafe { metadata.assume_init() };
        let expected = file.metadata()?;
        Ok(metadata.st_dev == expected.dev()
            && metadata.st_ino == expected.ino()
            && (metadata.st_mode & libc::S_IFMT) == libc::S_IFREG)
    }
    pub(super) fn verify(&self) -> PyResult<()> {
        let current = open_directory(&self.logical, false)?;
        let expected = self.file.metadata()?;
        let actual = current.file.metadata()?;
        if expected.dev() != actual.dev() || expected.ino() != actual.ino() {
            return Err(unsafe_path());
        }
        Ok(())
    }
    pub(super) fn unlink(&self, name: &std::ffi::OsStr, missing_ok: bool) -> PyResult<()> {
        let name = component(name)?;
        // SAFETY: unlinkat removes the named directory entry and never follows a final symlink.
        if unsafe { libc::unlinkat(self.file.as_raw_fd(), name.as_ptr(), 0) } < 0 {
            let error = io::Error::last_os_error();
            if !(missing_ok && error.kind() == io::ErrorKind::NotFound) {
                return Err(error.into());
            }
        }
        self.file.sync_all()?;
        Ok(())
    }
    pub(super) fn publish(
        &self,
        temporary: &std::ffi::OsStr,
        destination: &std::ffi::OsStr,
        file: &File,
    ) -> PyResult<()> {
        self.verify()?;
        if !self.same_file(temporary, file)? {
            return Err(unsafe_path());
        }
        let old = component(temporary)?;
        let new = component(destination)?;
        // SAFETY: both names are single components under this live directory; destination symlinks are replaced, not followed.
        if unsafe {
            libc::renameat(
                self.file.as_raw_fd(),
                old.as_ptr(),
                self.file.as_raw_fd(),
                new.as_ptr(),
            )
        } < 0
        {
            return Err(io::Error::last_os_error().into());
        }
        if !self.same_file(destination, file)? {
            return Err(unsafe_path());
        }
        self.verify()?;
        self.file.sync_all()?;
        Ok(())
    }
}
pub(super) fn descriptor_path(file: &File) -> PathBuf {
    PathBuf::from(format!(
        "/proc/{}/fd/{}",
        std::process::id(),
        file.as_raw_fd()
    ))
}
pub(super) fn open(path: &Path) -> PyResult<File> {
    let dir = parent(path, false)?;
    dir.open(path.file_name().ok_or_else(unsafe_path)?, libc::O_RDONLY)
}
pub(super) fn create_lock(path: &Path) -> PyResult<File> {
    let dir = parent(path, true)?;
    dir.open(
        path.file_name().ok_or_else(unsafe_path)?,
        libc::O_RDWR | libc::O_CREAT | libc::O_APPEND,
    )
}
pub(super) fn remove(path: &Path, missing_ok: bool) -> PyResult<()> {
    let dir = match parent(path, false) {
        Ok(dir) => dir,
        Err(error)
            if missing_ok
                && Python::attach(|py| {
                    error.is_instance_of::<pyo3::exceptions::PyFileNotFoundError>(py)
                }) =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    dir.unlink(path.file_name().ok_or_else(unsafe_path)?, missing_ok)
}
pub(super) fn read(path: &Path) -> PyResult<Vec<u8>> {
    let mut file = open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyTuple},
};
use std::collections::VecDeque;

struct Entry {
    key: Py<PyAny>,
    parts: Vec<Py<PyBytes>>,
    size: usize,
}
#[pyclass]
pub struct ByteResultCache {
    max_entries: usize,
    max_bytes: usize,
    entries: VecDeque<Entry>,
    bytes: usize,
}
#[pymethods]
impl ByteResultCache {
    #[new]
    #[pyo3(signature=(*,max_entries,max_bytes))]
    fn new(max_entries: isize, max_bytes: isize) -> PyResult<Self> {
        if max_entries < 1 {
            return Err(PyValueError::new_err(
                "ByteResultCache max_entries must be at least 1",
            ));
        }
        if max_bytes < 1 {
            return Err(PyValueError::new_err(
                "ByteResultCache max_bytes must be at least 1",
            ));
        }
        Ok(Self {
            max_entries: max_entries as usize,
            max_bytes: max_bytes as usize,
            entries: VecDeque::new(),
            bytes: 0,
        })
    }
    fn get(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        let Some(index) = self.index(key)? else {
            return Ok(None);
        };
        let entry = self.entries.remove(index).unwrap();
        let result = PyTuple::new(key.py(), entry.parts.iter().map(|p| p.bind(key.py())))?
            .into_any()
            .unbind();
        self.entries.push_back(entry);
        Ok(Some(result))
    }
    fn put(&mut self, key: &Bound<'_, PyAny>, parts: &Bound<'_, PyAny>) -> PyResult<()> {
        key.hash()?;
        let bytes = key.py().import("builtins")?.getattr("bytes")?;
        let parts: Vec<Py<PyBytes>> = parts
            .try_iter()?
            .map(|part| Ok(bytes.call1((part?,))?.cast_into::<PyBytes>()?.unbind()))
            .collect::<PyResult<_>>()?;
        let size = parts
            .iter()
            .map(|p| p.bind(key.py()).as_bytes().len())
            .sum();
        if let Some(i) = self.index(key)? {
            self.bytes -= self.entries.remove(i).unwrap().size;
        }
        if size > self.max_bytes {
            return Ok(());
        }
        self.entries.push_back(Entry {
            key: key.clone().unbind(),
            parts,
            size,
        });
        self.bytes += size;
        while self.entries.len() > self.max_entries || self.bytes > self.max_bytes {
            self.bytes -= self.entries.pop_front().unwrap().size;
        }
        Ok(())
    }
    fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}
impl ByteResultCache {
    fn index(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        key.hash()?;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.key.bind(key.py()).eq(key)? {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }
}
struct Handoff {
    key: Py<PyAny>,
    stamp: Py<PyAny>,
    blob: Vec<u8>,
}
#[pyclass]
pub struct Handoffs {
    #[pyo3(get)]
    limit: usize,
    entries: VecDeque<Handoff>,
    bytes: usize,
}
#[pymethods]
impl Handoffs {
    #[new]
    #[pyo3(signature=(max_bytes=268435456))]
    fn new(max_bytes: usize) -> Self {
        Self {
            limit: max_bytes,
            entries: VecDeque::new(),
            bytes: 0,
        }
    }
    #[getter]
    fn bytes_used(&self) -> usize {
        self.bytes
    }
    fn put(&mut self, key: &Bound<'_, PyAny>, stamp: Py<PyAny>, blob: Vec<u8>) -> PyResult<bool> {
        self.drop_key(key)?;
        if blob.len() > self.limit {
            return Ok(false);
        }
        self.bytes += blob.len();
        self.entries.push_back(Handoff {
            key: key.clone().unbind(),
            stamp,
            blob,
        });
        while self.bytes > self.limit {
            self.bytes -= self.entries.pop_front().unwrap().blob.len();
        }
        Ok(true)
    }
    fn take(
        &mut self,
        key: &Bound<'_, PyAny>,
        stamp: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyBytes>>> {
        let Some(entry) = self.drop_key(key)? else {
            return Ok(None);
        };
        if entry.stamp.bind(key.py()).eq(stamp)? {
            Ok(Some(PyBytes::new(key.py(), &entry.blob).unbind()))
        } else {
            Ok(None)
        }
    }
    fn retain(&mut self, prompt_ids: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut i = 0;
        while i < self.entries.len() {
            let key = self.entries[i].key.bind(prompt_ids.py()).get_item(0)?;
            if prompt_ids.contains(key)? {
                i += 1;
            } else {
                self.bytes -= self.entries.remove(i).unwrap().blob.len();
            }
        }
        Ok(())
    }
}
impl Handoffs {
    fn drop_key(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Option<Handoff>> {
        for i in 0..self.entries.len() {
            if self.entries[i].key.bind(key.py()).eq(key)? {
                let entry = self.entries.remove(i).unwrap();
                self.bytes -= entry.blob.len();
                return Ok(Some(entry));
            }
        }
        Ok(None)
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ByteResultCache>()?;
    m.add_class::<Handoffs>()?;
    Ok(())
}

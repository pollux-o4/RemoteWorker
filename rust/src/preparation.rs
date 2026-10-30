use pyo3::{prelude::*, types::PyDict};

#[derive(Debug, PartialEq, Eq)]
enum Decision {
    Prefetch,
    Hold,
    WaitProgress,
    WaitClip,
    Warmup,
}
impl Decision {
    fn decide(remaining: Option<i64>, needs_clip: bool, reserved: bool) -> Self {
        if needs_clip {
            Self::Prefetch
        } else if reserved {
            Self::Hold
        } else {
            match remaining {
                None => Self::WaitProgress,
                Some(n) if n > 2 => Self::WaitClip,
                Some(_) => Self::Warmup,
            }
        }
    }
    fn labels(&self) -> (&'static str, &'static str) {
        match self {
            Self::Prefetch => ("prefetch", "clip_required"),
            Self::Hold => ("hold", "vae_ready"),
            Self::WaitProgress => ("wait", "sampling_progress"),
            Self::WaitClip => ("wait", "clip_window"),
            Self::Warmup => ("warmup", "vae_window"),
        }
    }
}
#[pyfunction]
fn admission(
    remaining_steps: Option<i64>,
    needs_clip: bool,
    reserved: bool,
) -> (String, String, Option<i64>) {
    let (action, reason) = Decision::decide(remaining_steps, needs_clip, reserved).labels();
    (action.into(), reason.into(), remaining_steps)
}
#[pyclass]
pub struct SamplingProgress {
    #[pyo3(get)]
    job: Option<String>,
    #[pyo3(get)]
    node: Option<String>,
    #[pyo3(get)]
    value: i64,
    #[pyo3(get)]
    total: i64,
}
#[pymethods]
impl SamplingProgress {
    #[new]
    fn new() -> Self {
        Self {
            job: None,
            node: None,
            value: 0,
            total: 0,
        }
    }
    fn update(&mut self, job: String, node: String, value: i64, total: i64) {
        if total <= 0 || value < 0 || value > total || (total == 1 && value == 0) {
            return;
        }
        self.job = Some(job);
        self.node = Some(node);
        self.value = value;
        self.total = total;
    }
    fn reset(&mut self) {
        *self = Self::new();
    }
    fn detail<'py>(&self, py: Python<'py>, job: Option<String>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if job == self.job {
            dict.set_item("step", self.value)?;
            dict.set_item("total", self.total)?;
        }
        Ok(dict)
    }
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(admission, m)?)?;
    m.add_class::<SamplingProgress>()?;
    m.add_class::<Scheduler>()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clip_preempts_vae() {
        assert_eq!(Decision::decide(Some(0), true, true), Decision::Prefetch);
        assert_eq!(Decision::decide(None, false, true), Decision::Hold);
        assert_eq!(Decision::decide(Some(3), false, false), Decision::WaitClip);
        assert_eq!(Decision::decide(Some(2), false, false), Decision::Warmup);
        assert_eq!(Decision::decide(None, false, false), Decision::WaitProgress);
    }
}

use crate::{from_python, to_python};
use serde_json::json;
use std::collections::HashSet;
#[pyclass]
struct Scheduler {
    #[pyo3(get)]
    sampling: Py<SamplingProgress>,
    reservations: HashSet<(String, String, String)>,
    #[pyo3(get)]
    last: Py<PyDict>,
    counts: std::collections::HashMap<String, u64>,
    durations: std::collections::HashMap<String, f64>,
}
#[pymethods]
impl Scheduler {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            sampling: Py::new(py, SamplingProgress::new())?,
            reservations: HashSet::new(),
            last: PyDict::new(py).unbind(),
            counts: [("encoder".into(), 0), ("vae".into(), 0)].into(),
            durations: Default::default(),
        })
    }
    fn observe_encoder(&mut self, seconds: f64) {
        self.observe("encoder", seconds);
    }
    fn observe_vae(&mut self, seconds: f64) {
        self.observe("vae", seconds);
    }
    fn observe(&mut self, kind: &str, seconds: f64) {
        *self.counts.entry(kind.to_owned()).or_default() += 1;
        self.durations.insert(kind.to_owned(), seconds);
    }
    fn retain(&mut self, py: Python<'_>, ids: HashSet<String>) -> PyResult<()> {
        self.reservations.retain(|r| ids.contains(&r.0));
        let mut sampling = self.sampling.borrow_mut(py);
        if sampling.job.as_ref().is_none_or(|j| !ids.contains(j)) {
            sampling.reset();
        }
        let last = self.last.bind(py);
        if last
            .get_item("prompt_id")?
            .and_then(|v| v.extract::<String>().ok())
            .is_none_or(|j| !ids.contains(&j))
        {
            last.clear();
        }
        Ok(())
    }
    fn reserve(&mut self, job: String, decoder: (String, String, String, String)) {
        self.reservations
            .insert((job, decoder.1.trim_end_matches('/').into(), decoder.2));
    }
    fn invalidate_vae(&mut self, job: &str, url: &str) {
        self.reservations
            .retain(|r| r.0 != job || r.1 != url.trim_end_matches('/'));
    }
    fn reserved(&self, job: &str, decoder: Option<(String, String, String, String)>) -> bool {
        decoder.is_some_and(|d| {
            self.reservations
                .contains(&(job.into(), d.1.trim_end_matches('/').into(), d.2))
        })
    }
    #[pyo3(signature=(job,following=None,*,needs_clip=false,decoder=None))]
    fn decide(
        &mut self,
        py: Python<'_>,
        job: String,
        following: Option<String>,
        needs_clip: bool,
        decoder: Option<(String, String, String, String)>,
    ) -> PyResult<Py<PyAny>> {
        let sampling = self.sampling.borrow(py);
        let steps = if sampling.job.as_ref() == Some(&job) && sampling.total > 0 {
            Some((sampling.total - sampling.value).max(0))
        } else {
            None
        };
        let (action, reason) =
            Decision::decide(steps, needs_clip, self.reserved(&job, decoder)).labels();
        let last = PyDict::new(py);
        last.set_item("action", action)?;
        last.set_item("reason", reason)?;
        last.set_item("remaining_steps", steps)?;
        last.set_item("prompt_id", &job)?;
        last.set_item("next_prompt_id", following)?;
        if sampling.job.as_ref() == Some(&job) {
            last.set_item("step", sampling.value)?;
            last.set_item("total", sampling.total)?;
        }
        last.set_item("at", py.import("time")?.call_method0("time")?)?;
        self.last = last.unbind();
        let decision = py.import("types")?.getattr("SimpleNamespace")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("action", action)?;
        kwargs.set_item("reason", reason)?;
        kwargs.set_item("remaining_steps", steps)?;
        Ok(decision.call((), Some(&kwargs))?.unbind())
    }
    fn status(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let sample = self.sampling.borrow(py);
        let mut sampling = json!({"prompt_id":sample.job});
        sampling["step"] = sample.value.into();
        sampling["total"] = sample.total.into();
        to_python(
            py,
            &json!({"policy":"clip_first_two_steps","vae_remaining_steps":2,"decision":from_python(self.last.bind(py).as_any())?,"measured_samples":self.counts,"last_seconds":self.durations,"sampling":sampling}),
        )
    }
}

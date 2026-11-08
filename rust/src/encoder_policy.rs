mod cache;
mod media;
mod peer_io;
mod preparation;
mod vae;
mod validation;

use pyo3::{prelude::*, types::PyModule};
use validation::{clean_model_name_native, invalid, minimax_policy_native};
pub(crate) use validation::{
    qwen_resize_native, qwen_worker_inputs_native, validate_prompt_media_native,
};

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    validation::register_base(m)?;
    media::register(m)?;
    validation::register_qwen(m)?;
    vae::register(m)?;
    cache::register(m)?;
    peer_io::register(m)?;
    preparation::register(m)?;
    validation::register_local(m)?;
    Ok(())
}

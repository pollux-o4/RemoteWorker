"""Context-manager syntax adapter for Rust-owned request lifetimes."""
from contextlib import contextmanager
from .._native import (
    CompletedPrompt as completed_prompt,
    RemoteFailureScope as cancel_on_remote_failure,
    RequestInputs as RequestInputs,
    download_artifacts,
)
from .artifacts import extract_artifact_paths


@contextmanager
def completed_artifacts(api, workflow, request_id, specs):
    with completed_prompt(api, workflow) as (prompt_id, history):
        paths = extract_artifact_paths(history, prompt_id, request_id, specs)
        yield download_artifacts(api, paths)

__all__ = ["completed_prompt", "cancel_on_remote_failure", "RequestInputs", "completed_artifacts"]

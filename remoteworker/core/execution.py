"""One request lifecycle: artifact retrieval, targeted cancellation and input cleanup."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from .artifacts import extract_artifact_paths


@contextmanager
def cancel_on_remote_failure(api: Any, prompt_id: str):
    """Cancel one submitted peer prompt without masking the caller's failure."""
    try:
        yield
    except BaseException:
        try:
            api.cancel_prompt(prompt_id)
        except BaseException as exc:
            logging.getLogger(__name__).warning("RemoteWorker could not cancel remote prompt %s: %s", prompt_id, exc)
        raise


@contextmanager
def completed_artifacts(api, workflow, request_id, specs):
    """Keep cancellation active through the caller's result decoding."""
    prompt_id = api.submit(workflow)
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("RemoteWorker API returned no prompt_id")
    with cancel_on_remote_failure(api, prompt_id):
        paths = extract_artifact_paths(api.wait_for_history(prompt_id), prompt_id, request_id, specs)
        yield {name: api.download_artifact(path) for name, path in paths.items()}


class RequestInputs:
    """Track successful uploads and discard only this request's temporary inputs."""

    def __init__(self, api, request_id):
        self.api = api
        self.request_id = request_id
        self.uploaded = False

    def __enter__(self):
        return self

    def upload_image(self, **kwargs):
        result = self.api.upload_image(request_id=self.request_id, **kwargs)
        self.uploaded = True
        return result

    def upload_file(self, **kwargs):
        result = self.api.upload_file(request_id=self.request_id, **kwargs)
        self.uploaded = True
        return result

    def __exit__(self, exc_type, exc, traceback):
        if self.uploaded:
            try:
                self.api.discard_inputs(self.request_id)
            except (OSError, RuntimeError, TimeoutError, ValueError) as cleanup_error:
                logging.getLogger(__name__).warning("RemoteWorker could not discard request inputs: %s", cleanup_error)
        return False

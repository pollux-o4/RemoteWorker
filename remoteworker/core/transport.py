"""HTTP adapter for a ComfyUI peer; owns IO, polling and cancellation transport."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .. import _native

from .protocol import (
    _parse_env_positive_int,
    _validate_server_url,
    parse_clip_loader_options,
    parse_vae_loader_options,
)

_POLL_INTERVAL_SECONDS = 0.25
_MAX_ARTIFACT_BYTES = _parse_env_positive_int("REMOTE_CONDITIONING_MAX_ARTIFACT_BYTES", 2 * 1024 * 1024 * 1024)


class RemoteHttpError(RuntimeError):
    """HTTP failure with a status code for narrow endpoint negotiation."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        super().__init__(f"Remote Conditioning HTTP {status}: {detail}")


class HttpComfyApi:
    def __init__(self, server_url: str, timeout_seconds: int) -> None:
        self._server_url = _validate_server_url(server_url)
        self._timeout_seconds = timeout_seconds

    def _request(
        self,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str = "application/json",
    ) -> bytes:
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = content_type
        request = Request(
            f"{self._server_url}{path}",
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read(_MAX_ARTIFACT_BYTES + 1)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RemoteHttpError(exc.code, detail) from exc
        except URLError as exc:
            raise RuntimeError(f"Remote Conditioning peer connection failed: {exc.reason}") from exc
        _native.check_artifact_budget(len(body), _MAX_ARTIFACT_BYTES)
        return body

    def _json(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            result = json.loads(
                self._request(
                    path,
                    data=json.dumps(payload).encode("utf-8") if payload is not None else None,
                ).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Remote Conditioning returned invalid JSON from {path}") from exc
        _native.object_response(result, path)
        return result

    def submit(self, workflow: dict[str, Any]) -> str:
        from .cancellation import check_cancelled
        check_cancelled()
        response = self._json("/prompt", payload={"prompt": workflow, "client_id": uuid.uuid4().hex})
        return _native.prompt_response(response)

    def cancel_prompt(self, prompt_id: str) -> None:
        """Cancel a submitted peer prompt without affecting other peer work."""
        _native.validate_prompt_id(prompt_id)
        try:
            response = self._json(f"/api/jobs/{quote(prompt_id, safe='')}/cancel", payload={})
        except RemoteHttpError as exc:
            if exc.status != 404:
                raise
            self._request("/interrupt", data=json.dumps({"prompt_id": prompt_id}).encode("utf-8"))
            self._request("/queue", data=json.dumps({"delete": [prompt_id]}).encode("utf-8"))
            return
        _native.boolean_response(response, "cancelled", "Remote Conditioning prompt cancel returned an invalid response")

    def clip_loader_options(self) -> dict[str, list[str]]:
        """Return the peer's validated CLIPLoader choices."""
        return parse_clip_loader_options(self._json("/object_info/CLIPLoader"))

    def vae_loader_options(self) -> dict[str, list[str]]:
        """Return the peer's validated VAELoader choices."""
        return parse_vae_loader_options(self._json("/object_info/VAELoader"))

    def require_node(self, node_name: str) -> dict[str, Any]:
        info = self._json(f"/object_info/{node_name}")
        return _native.require_node_response(info, node_name)

    def upload_image(self, *, request_id: str, index: int, png: bytes) -> str:
        _native.upload_image_index(index)
        return self.upload_file(
            request_id=request_id,
            filename=f"image-{index:04d}.png",
            data=png,
            content_type="image/png",
        )

    def upload_file(self, *, request_id: str, filename: str, data: bytes, content_type: str) -> str:
        """Upload one request-scoped media file through ComfyUI's temp upload API."""
        boundary = f"RemoteConditioning{uuid.uuid4().hex}"
        body = _native.upload_multipart(request_id, filename, data, content_type, _MAX_ARTIFACT_BYTES, boundary)
        try:
            result = json.loads(
                self._request(
                    "/upload/image",
                    data=body,
                    content_type=f"multipart/form-data; boundary={boundary}",
                ).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Remote Conditioning media upload returned invalid JSON") from exc
        return _native.upload_response(result, request_id, filename)

    def discard_inputs(self, request_id: str) -> bool:
        response = self._json("/remote-conditioning/discard-inputs", payload={"request_id": request_id})
        return _native.boolean_response(response, "removed", "Remote Conditioning discard-inputs returned an invalid response")

    def wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            from .cancellation import check_cancelled
            check_cancelled()
            _raise_if_caller_interrupted()
            history = self._json(f"/history/{prompt_id}")
            if _native.completed_history(history, prompt_id):
                return history
            time.sleep(_POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"Remote Conditioning timed out after {self._timeout_seconds}s (prompt {prompt_id})")

    def discard_vae_outputs(self, request_id: str) -> bool:
        response = self._json("/remote-vae/discard-output", payload={"request_id": request_id})
        return _native.boolean_response(response, "removed", "Remote VAE discard-output returned an invalid response")

    def download_artifact(self, artifact: str) -> bytes:
        return self._request(_native.download_query(artifact))


def _raise_if_caller_interrupted() -> None:
    """Use ComfyUI's cancellation signal when running inside its executor."""
    from .cancellation import caller_interrupt_enabled
    if not caller_interrupt_enabled():
        return
    try:
        from comfy import model_management
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()

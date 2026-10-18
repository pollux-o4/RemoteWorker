"""HTTP adapter for a ComfyUI peer; owns IO, polling and cancellation transport."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .protocol import (
    _parse_env_positive_int,
    _validate_server_url,
    is_valid_request_id,
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
        if len(body) > _MAX_ARTIFACT_BYTES:
            limit_mb = _MAX_ARTIFACT_BYTES // (1024 * 1024)
            actual_mb = round(len(body) / (1024 * 1024), 2)
            raise RuntimeError(
                f"Remote Conditioning artifact exceeds the {limit_mb} MiB limit ({actual_mb} MiB received)"
            )
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
        if not isinstance(result, dict):
            raise RuntimeError(f"Remote Conditioning returned a non-object JSON response from {path}")
        return result

    def submit(self, workflow: dict[str, Any]) -> str:
        response = self._json("/prompt", payload={"prompt": workflow, "client_id": uuid.uuid4().hex})
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError(f"Remote Conditioning prompt was rejected: {response.get('error', response)}")
        return prompt_id

    def cancel_prompt(self, prompt_id: str) -> None:
        """Cancel a submitted peer prompt without affecting other peer work."""
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError("Remote Conditioning prompt_id must be a non-empty string")
        try:
            response = self._json(f"/api/jobs/{quote(prompt_id, safe='')}/cancel", payload={})
        except RemoteHttpError as exc:
            if exc.status != 404:
                raise
            self._request("/interrupt", data=json.dumps({"prompt_id": prompt_id}).encode("utf-8"))
            self._request("/queue", data=json.dumps({"delete": [prompt_id]}).encode("utf-8"))
            return
        if set(response) != {"cancelled"} or not isinstance(response["cancelled"], bool):
            raise RuntimeError("Remote Conditioning prompt cancel returned an invalid response")

    def clip_loader_options(self) -> dict[str, list[str]]:
        """Return the peer's validated CLIPLoader choices."""
        return parse_clip_loader_options(self._json("/object_info/CLIPLoader"))

    def vae_loader_options(self) -> dict[str, list[str]]:
        """Return the peer's validated VAELoader choices."""
        return parse_vae_loader_options(self._json("/object_info/VAELoader"))

    def require_node(self, node_name: str) -> dict[str, Any]:
        info = self._json(f"/object_info/{node_name}")
        node = info.get(node_name)
        if not isinstance(node, dict):
            raise RuntimeError(f"Remote Conditioning peer does not provide {node_name}")
        return node

    def upload_image(self, *, request_id: str, index: int, png: bytes) -> str:
        if not isinstance(index, int) or index < 1:
            raise ValueError("Remote Conditioning image upload arguments are invalid")
        return self.upload_file(
            request_id=request_id,
            filename=f"image-{index:04d}.png",
            data=png,
            content_type="image/png",
        )

    def upload_file(self, *, request_id: str, filename: str, data: bytes, content_type: str) -> str:
        """Upload one request-scoped media file through ComfyUI's temp upload API."""
        if not is_valid_request_id(request_id):
            raise ValueError("Remote Conditioning request_id must be 1-64 URL-safe characters")
        if (
            not isinstance(filename, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", filename)
            or not isinstance(content_type, str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+/-]{0,127}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+/-]{0,127}",
                content_type,
            )
            or not isinstance(data, bytes)
            or not data
            or len(data) > _MAX_ARTIFACT_BYTES
        ):
            raise ValueError("Remote Conditioning media upload arguments are invalid")
        boundary = f"RemoteConditioning{uuid.uuid4().hex}"
        subfolder = f"remote-conditioning/{request_id}"
        parts: list[bytes] = []
        for name, value in (
            ("type", "temp"),
            ("subfolder", subfolder),
            ("overwrite", "true"),
        ):
            parts.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode(),
                ]
            )
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                data,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        try:
            result = json.loads(
                self._request(
                    "/upload/image",
                    data=b"".join(parts),
                    content_type=f"multipart/form-data; boundary={boundary}",
                ).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Remote Conditioning media upload returned invalid JSON") from exc
        expected = {"name": filename, "subfolder": subfolder, "type": "temp"}
        if not isinstance(result, dict) or any(result.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Remote Conditioning media upload returned an unexpected path: {result!r}")
        return f"{subfolder}/{filename} [temp]"

    def discard_inputs(self, request_id: str) -> bool:
        response = self._json("/remote-conditioning/discard-inputs", payload={"request_id": request_id})
        if set(response) != {"removed"} or not isinstance(response["removed"], bool):
            raise RuntimeError("Remote Conditioning discard-inputs returned an invalid response")
        return response["removed"]

    def wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_seconds
        while time.monotonic() < deadline:
            _raise_if_caller_interrupted()
            history = self._json(f"/history/{prompt_id}")
            entry = history.get(prompt_id)
            status = entry.get("status", {}).get("status_str") if isinstance(entry, dict) else None
            if status in {"success", "error"}:
                return history
            time.sleep(_POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"Remote Conditioning timed out after {self._timeout_seconds}s (prompt {prompt_id})")

    def download_artifact(self, artifact: str) -> bytes:
        subfolder, filename = artifact.rsplit("/", 1)
        return self._request(f"/view?{urlencode({'filename': filename, 'subfolder': subfolder, 'type': 'output'})}")


def _raise_if_caller_interrupted() -> None:
    """Use ComfyUI's cancellation signal when running inside its executor."""
    try:
        from comfy import model_management
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()

"""Queue a dummy decode now; join its request-scoped receipt before real decode."""
from __future__ import annotations

import json
import uuid

from ..core.artifacts import PROMPT_ARTIFACT, extract_artifact_paths
from ..core.execution import cancel_on_remote_failure


def schedule(api, *, vae_name, kind, server_url):
    request_id = uuid.uuid4().hex
    graph = {
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "warmup": {"class_type": "RemoteVAEWarmupWorker", "inputs": {"vae": ["vae", 0], "kind": kind}},
        "save": {"class_type": "SaveRemoteText", "inputs": {
            "text": ["warmup", 0], "filename_prefix": f"remote-conditioning/{request_id}/warmup"}},
    }
    prompt_id = api.submit(graph)
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("Remote VAE warmup returned no prompt_id")
    return json.dumps(dict(prompt_id=prompt_id, request_id=request_id,
        server_url=server_url.rstrip("/"), vae_name=vae_name, kind=kind))


def finish(api, ticket, *, vae_name, kind, server_url):
    if ticket in (None, "ready"):
        return
    try:
        data = json.loads(ticket)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid Remote VAE warmup ticket") from exc
    fields = {"prompt_id", "request_id", "server_url", "vae_name", "kind"}
    if (not isinstance(data, dict) or set(data) != fields
            or not all(isinstance(v, str) and v for v in data.values())
            or (data["server_url"], data["vae_name"], data["kind"]) != (server_url.rstrip("/"), vae_name, kind)):
        raise ValueError("준비 신호와 처리 서버·VAE·대상이 다릅니다. 두 노드의 설정을 맞추세요.")
    with cancel_on_remote_failure(api, data["prompt_id"]):
        paths = extract_artifact_paths(api.wait_for_history(data["prompt_id"]),
            data["prompt_id"], data["request_id"], (PROMPT_ARTIFACT,))
        try:
            if api.download_artifact(paths["prompt"]) != b"ready":
                raise RuntimeError("Remote VAE warmup did not finish")
        finally:
            if hasattr(api, 'discard_vae_outputs'):
                try:
                    api.discard_vae_outputs(data['request_id'])
                except (OSError, RuntimeError, TimeoutError, ValueError):
                    import logging
                    logging.getLogger(__name__).warning('Could not discard warmup receipt')

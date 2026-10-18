"""Request-scoped remote ComfyUI output artifact contracts."""
from __future__ import annotations

from typing import Any, NamedTuple


class RemoteArtifactSpec(NamedTuple):
    """One output-node artifact that a remote strategy is allowed to recover."""

    name: str
    output_node: str
    ui_field: str
    extension: str


CONDITIONING_ARTIFACT = RemoteArtifactSpec("conditioning", "save", "text", ".cond")
PROMPT_ARTIFACT = RemoteArtifactSpec("prompt", "save", "text", ".txt")


def _raise_peer_execution_error(history: dict[str, Any], prompt_id: str) -> None:
    entry = history[prompt_id]
    messages = entry.get("status", {}).get("messages", [])
    for message in reversed(messages) if isinstance(messages, list) else ():
        if not isinstance(message, list) or len(message) != 2 or message[0] != "execution_error":
            continue
        detail = message[1]
        if not isinstance(detail, dict):
            continue
        node_type = detail.get("node_type")
        exception_type = detail.get("exception_type")
        exception_message = detail.get("exception_message")
        if all(isinstance(value, str) and value.strip() for value in (node_type, exception_type, exception_message)):
            concise_message = " ".join(exception_message.split())[:500]
            raise RuntimeError(
                f"Remote Conditioning peer execution failed at {node_type}: "
                f"{exception_type}: {concise_message} (prompt {prompt_id})"
            )
    raise RuntimeError(f"Remote Conditioning peer execution failed for prompt {prompt_id}")


def _path_from_locator(locator: Any, spec: RemoteArtifactSpec) -> str:
    if isinstance(locator, str):
        return locator
    if not isinstance(locator, dict):
        raise RuntimeError(f"Remote Conditioning {spec.name} artifact locator is invalid")
    filename = locator.get("filename")
    subfolder = locator.get("subfolder")
    artifact_type = locator.get("type")
    if (
        not isinstance(filename, str)
        or not isinstance(subfolder, str)
        or artifact_type != "output"
    ):
        raise RuntimeError(f"Remote Conditioning {spec.name} artifact locator is invalid")
    return f"{subfolder}/{filename}" if subfolder else filename


def _validate_artifact_path(artifact: str, request_id: str, spec: RemoteArtifactSpec) -> str:
    expected_prefix = f"remote-conditioning/{request_id}/"
    parts = artifact.split("/")
    if (
        not artifact.startswith(expected_prefix)
        or "\\" in artifact
        or any(part in {"", ".", ".."} for part in parts)
        or not artifact.endswith(spec.extension)
    ):
        raise RuntimeError(f"Remote Conditioning {spec.name} artifact is outside its request directory: {artifact!r}")
    return artifact


def extract_artifact_paths(
    history: dict[str, Any],
    prompt_id: str,
    request_id: str,
    specs: tuple[RemoteArtifactSpec, ...],
) -> dict[str, str]:
    """Recover exactly one validated output artifact for each declared strategy result."""
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Remote Conditioning artifact specs must be non-empty with unique names")
    try:
        entry = history[prompt_id]
        status = entry["status"]["status_str"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Remote Conditioning history is malformed for prompt {prompt_id}") from exc
    if status == "error":
        _raise_peer_execution_error(history, prompt_id)
    if status != "success":
        raise RuntimeError(f"Remote Conditioning is not complete for prompt {prompt_id}: {status!r}")

    artifacts: dict[str, str] = {}
    for spec in specs:
        try:
            locators = entry["outputs"][spec.output_node][spec.ui_field]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Remote Conditioning {spec.name} output is missing for prompt {prompt_id}") from exc
        if not isinstance(locators, list) or len(locators) != 1:
            raise RuntimeError(f"Remote Conditioning must produce exactly one {spec.name} artifact for prompt {prompt_id}")
        artifact = _path_from_locator(locators[0], spec)
        artifacts[spec.name] = _validate_artifact_path(artifact, request_id, spec)
    return artifacts

"""Request-scoped remote ComfyUI output artifact contracts."""

from __future__ import annotations

from typing import NamedTuple

from .._native import (
    extract_artifact_paths as extract_artifact_paths,
)
from .._native import (
    successful_outputs as successful_outputs,
)


class RemoteArtifactSpec(NamedTuple):
    """One output-node artifact that a remote strategy is allowed to recover."""

    name: str
    output_node: str
    ui_field: str
    extension: str


REMOTE_CONDITIONING_ARTIFACT = RemoteArtifactSpec("conditioning", "save", "conditioning", ".cond")
CONDITIONING_ARTIFACT = RemoteArtifactSpec("conditioning", "save", "text", ".cond")
PROMPT_ARTIFACT = RemoteArtifactSpec("prompt", "save", "text", ".txt")

"""peer-owned prompt-enhancer profile registry.

Profiles are deliberately small declarative records.  The caller graph selects a
profile id; only the peer registry resolves that id into a CLIP loader recipe.
This keeps model paths and prompt-generation limits out of the caller's trust
boundary while still allowing users to add, inspect, and delete profiles.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..defaults import LOCAL_SETTINGS

_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_VERSION = 1
_MAX_DESCRIPTION_LENGTH = 480
_MAX_GENERATION_LENGTH = 2048
_MEDIA_TYPES = frozenset({"image", "video"})


@dataclass(frozen=True)
class PromptEnhancerProfile:
    id: str
    description: str
    clip_name: str
    clip_type: str
    device: str
    max_length: int
    system_prompt: str
    accepted_media: tuple[str, ...]

    def as_public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["accepted_media"] = list(self.accepted_media)
        return result


DEFAULT_PROFILE = PromptEnhancerProfile(
    id="example",
    description="Example prompt profile",
    clip_name="text-encoder.safetensors",
    clip_type="krea2",
    device="default",
    max_length=256,
    system_prompt="Return a concise visual prompt.",
    accepted_media=("image",),
)


def validate_profile_payload(payload: Any) -> PromptEnhancerProfile:
    from .._native import validate_profile_payload_native

    values = validate_profile_payload_native(
        payload, DEFAULT_PROFILE.system_prompt, list(DEFAULT_PROFILE.accepted_media)
    )
    values["accepted_media"] = tuple(values["accepted_media"])
    return PromptEnhancerProfile(**values)


if "prompt_profile" in LOCAL_SETTINGS:
    DEFAULT_PROFILE = validate_profile_payload(LOCAL_SETTINGS["prompt_profile"])


def _default_storage_path() -> Path:
    try:
        import folder_paths

        return (
            Path(folder_paths.get_user_directory())
            / "remote-prompt-enhancer-profiles.json"
        )
    except (ImportError, AttributeError):
        return (
            Path(__file__).resolve().parents[2] / "remote-prompt-enhancer-profiles.json"
        )


class PromptEnhancerProfileStore:
    """A tiny peer-local repository with atomic saves and strict schema checks."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_storage_path()

    def list(self) -> list[PromptEnhancerProfile]:
        if not self.path.exists():
            return [DEFAULT_PROFILE]
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Prompt enhancer profile registry is unreadable: {self.path}"
            ) from exc
        from .._native import profile_registry_native

        records = profile_registry_native(
            document,
            DEFAULT_PROFILE.system_prompt,
            list(DEFAULT_PROFILE.accepted_media),
        )
        return [validate_profile_payload(record) for record in records]

    def get(self, profile_id: str) -> PromptEnhancerProfile:
        from .._native import validate_profile_id_native, profile_select_native

        validate_profile_id_native(profile_id)
        return profile_select_native(self.list(), profile_id)

    def upsert(self, payload: Any) -> PromptEnhancerProfile:
        from .._native import profile_mutate_native

        profile = validate_profile_payload(payload)
        profiles, _ = profile_mutate_native(self.list(), profile, profile.id)
        self._write(profiles)
        return profile

    def delete(self, profile_id: str) -> bool:
        from .._native import validate_profile_id_native, profile_mutate_native

        validate_profile_id_native(profile_id)
        kept, changed = profile_mutate_native(self.list(), None, profile_id)
        if changed:
            self._write(kept)
        return changed

    def _write(self, profiles: list[PromptEnhancerProfile]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                {
                    "version": _VERSION,
                    "profiles": [profile.as_public_dict() for profile in profiles],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".remote-prompt-profiles-", suffix=".json", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
            Path(temporary_name).replace(self.path)
        except OSError:
            Path(temporary_name).unlink(missing_ok=True)
            raise


__all__ = [
    "DEFAULT_PROFILE",
    "PromptEnhancerProfile",
    "PromptEnhancerProfileStore",
    "validate_profile_payload",
]

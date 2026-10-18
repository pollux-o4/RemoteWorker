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
from typing import Any, Mapping

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
    id="example", description="Example prompt profile", clip_name="text-encoder.safetensors",
    clip_type="krea2", device="default", max_length=256,
    system_prompt="Return a concise visual prompt.", accepted_media=("image",),
)


def validate_profile_payload(payload: Any) -> PromptEnhancerProfile:
    """Normalize one persisted or user-submitted profile and reject ambiguity."""
    legacy_fields = {"id", "description", "clip_name", "clip_type", "device", "max_length"}
    current_fields = legacy_fields | {"system_prompt", "accepted_media"}
    payload_fields = set(payload) if isinstance(payload, Mapping) else set()
    if not isinstance(payload, Mapping) or payload_fields not in (legacy_fields, current_fields):
        raise ValueError("Prompt enhancer profile must contain id, description, clip_name, clip_type, device, max_length, system_prompt, and accepted_media")
    values = {name: payload[name] for name in payload}
    # Existing single-user registries predate multimodal profiles.  Reading one
    # remains safe and the next write upgrades it to the complete schema.
    values.setdefault("system_prompt", DEFAULT_PROFILE.system_prompt)
    values.setdefault("accepted_media", list(DEFAULT_PROFILE.accepted_media))
    if not isinstance(values["id"], str) or not _PROFILE_ID.fullmatch(values["id"]):
        raise ValueError("Prompt enhancer profile id must be 1-64 lowercase letters, digits, or hyphens")
    for name in ("description", "clip_name", "clip_type", "device"):
        value = values[name]
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"Prompt enhancer profile {name} must be a non-empty string")
    if len(values["description"].strip()) > _MAX_DESCRIPTION_LENGTH:
        raise ValueError(f"Prompt enhancer profile description must be at most {_MAX_DESCRIPTION_LENGTH} characters")
    if not isinstance(values["system_prompt"], str) or not values["system_prompt"].strip() or "\x00" in values["system_prompt"]:
        raise ValueError("Prompt enhancer profile system_prompt must be a non-empty string")
    raw_media = values["accepted_media"]
    if not isinstance(raw_media, list) or not raw_media or any(not isinstance(item, str) for item in raw_media):
        raise ValueError("Prompt enhancer profile accepted_media must be a non-empty array")
    accepted_media = tuple(dict.fromkeys(item.strip() for item in raw_media))
    if not accepted_media or any(item not in _MEDIA_TYPES for item in accepted_media):
        raise ValueError("Prompt enhancer profile accepted_media supports only image and video")
    if "video" in accepted_media and values["clip_type"].strip() != "minimax":
        raise ValueError("Prompt enhancer profile video support requires clip_type minimax")
    max_length = values["max_length"]
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 16 <= max_length <= _MAX_GENERATION_LENGTH:
        raise ValueError(f"Prompt enhancer profile max_length must be 16 to {_MAX_GENERATION_LENGTH}")
    return PromptEnhancerProfile(
        id=values["id"],
        description=values["description"].strip(),
        clip_name=values["clip_name"].strip(),
        clip_type=values["clip_type"].strip(),
        device=values["device"].strip(),
        max_length=max_length,
        system_prompt=values["system_prompt"].strip(),
        accepted_media=accepted_media,
    )


if "prompt_profile" in LOCAL_SETTINGS:
    DEFAULT_PROFILE = validate_profile_payload(LOCAL_SETTINGS["prompt_profile"])


def _default_storage_path() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.get_user_directory()) / "remote-prompt-enhancer-profiles.json"
    except (ImportError, AttributeError):
        return Path(__file__).resolve().parents[2] / "remote-prompt-enhancer-profiles.json"


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
            raise RuntimeError(f"Prompt enhancer profile registry is unreadable: {self.path}") from exc
        if not isinstance(document, dict) or set(document) != {"version", "profiles"} or document["version"] != _VERSION:
            raise RuntimeError("Prompt enhancer profile registry has an unsupported schema")
        raw_profiles = document["profiles"]
        if not isinstance(raw_profiles, list):
            raise RuntimeError("Prompt enhancer profile registry profiles must be an array")
        profiles = [validate_profile_payload(profile) for profile in raw_profiles]
        if len({profile.id for profile in profiles}) != len(profiles):
            raise RuntimeError("Prompt enhancer profile registry contains duplicate ids")
        return sorted(profiles, key=lambda profile: profile.id)

    def get(self, profile_id: str) -> PromptEnhancerProfile:
        if not isinstance(profile_id, str) or not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError("Prompt enhancer profile_id is invalid")
        for profile in self.list():
            if profile.id == profile_id:
                return profile
        raise ValueError(f"Prompt enhancer profile is not registered: {profile_id}")

    def upsert(self, payload: Any) -> PromptEnhancerProfile:
        profile = validate_profile_payload(payload)
        profiles = {item.id: item for item in self.list()}
        profiles[profile.id] = profile
        self._write(sorted(profiles.values(), key=lambda item: item.id))
        return profile

    def delete(self, profile_id: str) -> bool:
        if not isinstance(profile_id, str) or not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError("Prompt enhancer profile_id is invalid")
        profiles = self.list()
        kept = [profile for profile in profiles if profile.id != profile_id]
        if len(kept) == len(profiles):
            return False
        self._write(kept)
        return True

    def _write(self, profiles: list[PromptEnhancerProfile]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": _VERSION, "profiles": [profile.as_public_dict() for profile in profiles]},
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".remote-prompt-profiles-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
            Path(temporary_name).replace(self.path)
        except OSError:
            Path(temporary_name).unlink(missing_ok=True)
            raise


__all__ = ["DEFAULT_PROFILE", "PromptEnhancerProfile", "PromptEnhancerProfileStore", "validate_profile_payload"]

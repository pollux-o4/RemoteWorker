"""Public defaults with optional machine-local overrides outside the repository."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_local_settings() -> dict[str, Any]:
    location = os.environ.get("REMOTEWORKER_CONFIG")
    if location == "":
        return {}
    path = Path(location).expanduser() if location else Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "remoteworker" / "defaults.json"
    if not path.exists():
        if location:
            raise RuntimeError("Configured RemoteWorker settings file does not exist")
        return {}
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("RemoteWorker local settings could not be read") from error
    fields = {"server_url", "conditioning_clip", "conditioning_clip_type", "qwen_clip", "qwen_vae", "minimax_clip", "minimax_video_vae", "minimax_audio_vae", "prompt_profile"}
    if not isinstance(settings, dict) or set(settings) - fields:
        raise ValueError("RemoteWorker local settings contain unsupported fields")
    if any(not isinstance(value, str) for key, value in settings.items() if key != "prompt_profile"):
        raise ValueError("RemoteWorker default overrides must be strings")
    if "prompt_profile" in settings and not isinstance(settings["prompt_profile"], dict):
        raise ValueError("RemoteWorker prompt_profile override must be an object")
    return settings


LOCAL_SETTINGS = load_local_settings()


def default_value(name: str, public_default: str) -> str:
    return LOCAL_SETTINGS.get(name, public_default)


DEFAULT_SERVER_URL = default_value("server_url", "http://127.0.0.1:8188")

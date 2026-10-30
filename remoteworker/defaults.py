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
    path = (
        Path(location).expanduser()
        if location
        else Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "remoteworker"
        / "defaults.json"
    )
    if not path.exists():
        if location:
            raise RuntimeError("Configured RemoteWorker settings file does not exist")
        return {}
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("RemoteWorker local settings could not be read") from error
    from ._native import validate_defaults_native

    return validate_defaults_native(settings)


LOCAL_SETTINGS = load_local_settings()


def default_value(name: str, public_default: str) -> str:
    return LOCAL_SETTINGS.get(name, public_default)


DEFAULT_SERVER_URL = default_value("server_url", "http://127.0.0.1:8188")

"""Remove consumed VAE transfer tensors, restricted to one request directory."""
from __future__ import annotations

import asyncio
from pathlib import Path
import re

from ..core.protocol import is_valid_request_id

ROUTE = "/remote-vae/discard-output"


def discard_output(request_id, output_root=None):
    if not is_valid_request_id(request_id):
        raise ValueError("Invalid Remote VAE request ID")
    if output_root is None:
        import folder_paths
        output_root = folder_paths.get_output_directory()
    root = Path(output_root).resolve()
    path = root / "remote-conditioning" / request_id
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError("Remote VAE output directory escapes output storage")
    removed = False
    if path.exists():
        for file in path.iterdir():
            if re.fullmatch(r"(?:decoded_[0-9]{5,}_\.pt|warmup_[0-9]{5,}_\.txt)", file.name) and file.is_file() and not file.is_symlink():
                file.unlink()
                removed = True
        if not any(path.iterdir()):
            path.rmdir()
    return removed


async def discard(request):
    from aiohttp import web
    try:
        data = await request.json()
        if not isinstance(data, dict) or set(data) != {"request_id"}:
            raise ValueError("Expected only request_id")
        removed = await asyncio.to_thread(discard_output, data["request_id"])
        return web.json_response({"removed": removed})
    except (OSError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


def register_routes():
    try:
        from server import PromptServer
    except ImportError:
        return
    if getattr(PromptServer, "instance", None) is not None:
        PromptServer.instance.routes.post(ROUTE)(discard)

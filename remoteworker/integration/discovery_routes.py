"""Same-origin ComfyUI route for Remote Conditioning CLIP capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import _native
from ..core.transport import HttpComfyApi

ROUTE_PATH = "/remote-conditioning/clip-loader-options"
VAE_ROUTE_PATH = "/remote-conditioning/vae-loader-options"
DISCARD_INPUTS_ROUTE_PATH = "/remote-conditioning/discard-inputs"
_REQUEST_FIELDS = {"server_url", "timeout_seconds"}
_OPTION_FIELDS = ("clip_name", "clip_type", "device")


def _error(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def _validate_payload(payload: Any) -> tuple[str, int]:
    return _native.proxy_config(payload)


def _validate_options(options: Any) -> dict[str, list[str]]:
    return _native.proxy_options(options, ["clip_name", "clip_type", "device"])


def _validate_vae_options(options: Any) -> dict[str, list[str]]:
    return _native.proxy_options(options, ["vae_name"])


def create_clip_loader_options_handler(
    api_factory: Callable[[str, int], Any] = HttpComfyApi,
    json_response: Callable[..., Any] | None = None,
) -> Callable[[Any], Any]:
    """Create the POST handler with injectable HTTP boundaries for unit tests."""

    def respond(payload: Any, *, status: int = 200) -> Any:
        if json_response is not None:
            return json_response(payload, status=status)
        from aiohttp import web

        return web.json_response(payload, status=status)

    async def clip_loader_options(request: Any) -> Any:
        try:
            payload = await request.json()
            server_url, timeout_seconds = _validate_payload(payload)
        except Exception as exc:
            return respond(_error("invalid_request", str(exc)), status=400)
        try:
            api = api_factory(server_url, timeout_seconds)
        except ValueError as exc:
            return respond(_error("invalid_request", str(exc)), status=400)
        try:
            options = _validate_options(await asyncio.to_thread(api.clip_loader_options))
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            return respond(_error("upstream_failure", str(exc)), status=502)
        return respond(options)

    return clip_loader_options


def create_vae_loader_options_handler(
    api_factory: Callable[[str, int], Any] = HttpComfyApi,
    json_response: Callable[..., Any] | None = None,
) -> Callable[[Any], Any]:
    """Create the POST handler for peer VAELoader choices."""

    def respond(payload: Any, *, status: int = 200) -> Any:
        if json_response is not None:
            return json_response(payload, status=status)
        from aiohttp import web

        return web.json_response(payload, status=status)

    async def vae_loader_options(request: Any) -> Any:
        try:
            payload = await request.json()
            server_url, timeout_seconds = _validate_payload(payload)
        except Exception as exc:
            return respond(_error("invalid_request", str(exc)), status=400)
        try:
            api = api_factory(server_url, timeout_seconds)
        except ValueError as exc:
            return respond(_error("invalid_request", str(exc)), status=400)
        try:
            options = _validate_vae_options(await asyncio.to_thread(api.vae_loader_options))
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            return respond(_error("upstream_failure", str(exc)), status=502)
        return respond(options)

    return vae_loader_options


def discard_request_inputs(request_id: str, temp_directory: str | Path | None = None) -> bool:
    if temp_directory is None:
        import folder_paths

        temp_directory = folder_paths.get_temp_directory()
    return _native.discard_request_inputs_native(request_id, temp_directory)


def create_discard_inputs_handler(
    json_response: Callable[..., Any] | None = None,
) -> Callable[[Any], Any]:
    def respond(payload: Any, *, status: int = 200) -> Any:
        if json_response is not None:
            return json_response(payload, status=status)
        from aiohttp import web

        return web.json_response(payload, status=status)

    async def discard_inputs(request: Any) -> Any:
        try:
            payload = await request.json()
            request_id = _native.validate_request_body(payload)
            removed = await asyncio.to_thread(discard_request_inputs, request_id)
        except (OSError, ValueError) as exc:
            return respond(_error("invalid_request", str(exc)), status=400)
        return respond({"removed": removed})

    return discard_inputs


def register_routes(prompt_server: Any | None = None) -> bool:
    """Register against ComfyUI when its server module is available."""
    if prompt_server is None:
        try:
            from server import PromptServer
        except ImportError:
            return False
        prompt_server = PromptServer
    if getattr(prompt_server, "instance", None) is None:
        return False
    prompt_server.instance.routes.post(ROUTE_PATH)(create_clip_loader_options_handler())
    prompt_server.instance.routes.post(VAE_ROUTE_PATH)(create_vae_loader_options_handler())
    prompt_server.instance.routes.post(DISCARD_INPUTS_ROUTE_PATH)(create_discard_inputs_handler())
    return True


__all__ = [
    "ROUTE_PATH",
    "VAE_ROUTE_PATH",
    "DISCARD_INPUTS_ROUTE_PATH",
    "create_clip_loader_options_handler",
    "create_vae_loader_options_handler",
    "create_discard_inputs_handler",
    "discard_request_inputs",
    "register_routes",
]

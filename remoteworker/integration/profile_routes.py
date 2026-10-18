"""peer-owned CRUD route for remote prompt-enhancer profiles.

The browser always calls the caller same-origin proxy for discovery.  Graph nodes
call this peer route directly, so a client cannot mutate another runtime's
registry merely by changing local UI state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ..prompt.profiles import PromptEnhancerProfileStore, validate_profile_payload
from ..prompt.transport import PromptComfyApi

PROFILES_ROUTE_PATH = "/remote-prompt-enhancer/profiles"
PROFILE_OPTIONS_PROXY_ROUTE_PATH = "/remote-prompt-enhancer/profile-options"
PROFILE_MANAGE_PROXY_ROUTE_PATH = "/remote-prompt-enhancer/profiles/manage"


def _error(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def _public_profiles(store: PromptEnhancerProfileStore) -> dict[str, list[dict[str, Any]]]:
    return {"profiles": [profile.as_public_dict() for profile in store.list()]}


def _validate_local_profile_availability(profile: Any) -> None:
    """On peer, reject registrations that cannot be loaded by its current CLIPLoader."""
    try:
        import folder_paths
        import nodes
    except ImportError:  # Standalone unit tests have no ComfyUI runtime.
        return
    available_names = folder_paths.get_filename_list("text_encoders")
    available_types = nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0]
    available_devices = nodes.CLIPLoader.INPUT_TYPES()["optional"]["device"][0]
    if profile.clip_name not in available_names:
        raise ValueError(f"Prompt enhancer profile clip_name is not available on this peer runtime: {profile.clip_name}")
    if profile.clip_type not in available_types:
        raise ValueError(f"Prompt enhancer profile clip_type is not supported on this peer runtime: {profile.clip_type}")
    if profile.device not in available_devices:
        raise ValueError(f"Prompt enhancer profile device is not supported on this peer runtime: {profile.device}")


def create_profiles_handler(
    store_factory: Callable[[], PromptEnhancerProfileStore] = PromptEnhancerProfileStore,
    json_response: Callable[..., Any] | None = None,
) -> Callable[[Any], Any]:
    """Create GET/list and POST/upsert-or-delete handler with injectable storage."""
    def respond(payload: Any, *, status: int = 200) -> Any:
        if json_response is not None:
            return json_response(payload, status=status)
        from aiohttp import web
        return web.json_response(payload, status=status)

    async def profiles(request: Any) -> Any:
        try:
            store = store_factory()
            if request.method == "GET":
                return respond(await asyncio.to_thread(_public_profiles, store))
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            action = payload.get("action")
            if action == "upsert" and set(payload) == {"action", "profile"}:
                profile = validate_profile_payload(payload["profile"])
                _validate_local_profile_availability(profile)
                stored = await asyncio.to_thread(store.upsert, profile.as_public_dict())
                return respond({"profile": stored.as_public_dict()})
            if action == "delete" and set(payload) == {"action", "profile_id"}:
                deleted = await asyncio.to_thread(store.delete, payload["profile_id"])
                return respond({"deleted": deleted})
            raise ValueError("request action must be upsert with profile or delete with profile_id")
        except (OSError, RuntimeError, ValueError) as exc:
            return respond(_error("invalid_request", str(exc)), status=400)

    return profiles


def _validate_proxy_payload(payload: Any) -> tuple[str, int]:
    if not isinstance(payload, dict) or set(payload) != {"server_url", "timeout_seconds"}:
        raise ValueError("request body must contain only server_url and timeout_seconds")
    server_url = payload["server_url"]
    timeout_seconds = payload["timeout_seconds"]
    if not isinstance(server_url, str) or not server_url.strip():
        raise ValueError("server_url must be a non-empty string")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 5 <= timeout_seconds <= 30:
        raise ValueError("timeout_seconds must be between 5 and 30")
    return server_url, timeout_seconds


def create_profile_options_proxy_handler(
    api_factory: Callable[[str, int], Any] = PromptComfyApi,
    json_response: Callable[..., Any] | None = None,
) -> Callable[[Any], Any]:
    """Expose peer profile descriptions to the caller-hosted frontend without CORS."""
    def respond(payload: Any, *, status: int = 200) -> Any:
        if json_response is not None:
            return json_response(payload, status=status)
        from aiohttp import web
        return web.json_response(payload, status=status)

    async def options(request: Any) -> Any:
        try:
            server_url, timeout_seconds = _validate_proxy_payload(await request.json())
            profiles = await asyncio.to_thread(api_factory(server_url, timeout_seconds).list_prompt_enhancer_profiles)
            normalized = [validate_profile_payload(profile).as_public_dict() for profile in profiles]
            return respond({"profiles": normalized})
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            return respond(_error("upstream_failure", str(exc)), status=502)

    return options


def create_profile_manage_proxy_handler(
    api_factory: Callable[[str, int], Any] = PromptComfyApi,
    json_response: Callable[..., Any] | None = None,
) -> Callable[[Any], Any]:
    """Apply explicit profile CRUD from caller's same-origin management dialog."""
    def respond(payload: Any, *, status: int = 200) -> Any:
        if json_response is not None:
            return json_response(payload, status=status)
        from aiohttp import web
        return web.json_response(payload, status=status)

    async def manage(request: Any) -> Any:
        try:
            payload = await request.json()
            server_url, timeout_seconds = _validate_proxy_payload({
                "server_url": payload.get("server_url") if isinstance(payload, dict) else None,
                "timeout_seconds": payload.get("timeout_seconds") if isinstance(payload, dict) else None,
            })
            if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
                raise ValueError("request action is required")
            api = api_factory(server_url, timeout_seconds)
            if payload["action"] == "upsert" and set(payload) == {"server_url", "timeout_seconds", "action", "profile"}:
                profile = validate_profile_payload(payload["profile"])
                stored = await asyncio.to_thread(api.upsert_prompt_enhancer_profile, profile.as_public_dict())
                return respond({"profile": stored})
            if payload["action"] == "delete" and set(payload) == {"server_url", "timeout_seconds", "action", "profile_id"}:
                deleted = await asyncio.to_thread(api.delete_prompt_enhancer_profile, payload["profile_id"])
                return respond({"deleted": deleted})
            raise ValueError("request action must be upsert with profile or delete with profile_id")
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            return respond(_error("upstream_failure", str(exc)), status=502)

    return manage


def register_routes(prompt_server: Any | None = None) -> bool:
    if prompt_server is None:
        try:
            from server import PromptServer
        except ImportError:
            return False
        prompt_server = PromptServer
    prompt_server.instance.routes.get(PROFILES_ROUTE_PATH)(create_profiles_handler())
    if getattr(prompt_server, "instance", None) is None:
        return False
    prompt_server.instance.routes.post(PROFILES_ROUTE_PATH)(create_profiles_handler())
    prompt_server.instance.routes.post(PROFILE_OPTIONS_PROXY_ROUTE_PATH)(create_profile_options_proxy_handler())
    prompt_server.instance.routes.post(PROFILE_MANAGE_PROXY_ROUTE_PATH)(create_profile_manage_proxy_handler())
    return True




__all__ = [
    "PROFILES_ROUTE_PATH",
    "PROFILE_OPTIONS_PROXY_ROUTE_PATH",
    "PROFILE_MANAGE_PROXY_ROUTE_PATH",
    "create_profiles_handler",
    "create_profile_options_proxy_handler",
    "create_profile_manage_proxy_handler",
    "register_routes",
]

"""Remove consumed VAE transfer tensors, restricted to one request directory."""

from __future__ import annotations

import asyncio
from pathlib import Path

from .. import _native

ROUTE = "/remote-vae/discard-output"


def discard_output(request_id, output_root=None):
    if output_root is None:
        import folder_paths

        output_root = folder_paths.get_output_directory()
    from .._native import vae_discard_output

    return vae_discard_output(request_id, output_root)


async def discard(request):
    from aiohttp import web

    try:
        data = _native.vae_request_body(await request.json(), ["request_id"], "Expected only request_id")
        removed = await asyncio.to_thread(discard_output, data["request_id"])
        return web.json_response({"removed": removed})
    except (OSError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def shared_check(request):
    from aiohttp import web

    from .delivery import receive_probe

    try:
        data = _native.vae_request_body(
            await request.json(),
            ["request_id", "token"],
            "Invalid shared storage challenge",
        )
        return web.json_response(await asyncio.to_thread(receive_probe, **data))
    except (OSError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def preflight(request):
    from aiohttp import web

    from .delivery import check_shared
    from .service import peer_api

    try:
        data = _native.vae_request_body(await request.json(), ["server_url"], "Expected server_url")
        return web.json_response(await asyncio.to_thread(check_shared, peer_api(data["server_url"], 5)))
    except (OSError, ValueError, RuntimeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def media(request):
    from aiohttp import web

    from .shared_store import read_receipt

    try:
        receipt = await asyncio.to_thread(read_receipt, request.match_info["request_id"])
        if receipt is None:
            raise FileNotFoundError("공유 결과 파일이 없습니다.")
        if not Path(receipt["path"]).is_file():
            raise FileNotFoundError("결합된 최종 영상에서 확인하세요.")
        return web.FileResponse(receipt["path"], headers={"Cache-Control": "no-store"})
    except (OSError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=404)


def register_routes():
    try:
        from server import PromptServer
    except ImportError:
        return
    if getattr(PromptServer, "instance", None) is not None:
        routes = PromptServer.instance.routes
        routes.post(ROUTE)(discard)
        routes.post("/remote-vae/shared/check")(shared_check)
        routes.post("/remote-vae/preflight")(preflight)
        routes.get("/remote-vae/media/{request_id}")(media)

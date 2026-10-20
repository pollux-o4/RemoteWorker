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


async def shared_check(request):
    from aiohttp import web
    from .delivery import receive_probe
    try:
        data = await request.json()
        if not isinstance(data, dict) or set(data) != {'request_id', 'token'}:
            raise ValueError('Invalid shared storage challenge')
        return web.json_response(await asyncio.to_thread(receive_probe, **data))
    except (OSError, ValueError) as exc:
        return web.json_response({'error': str(exc)}, status=400)


async def preflight(request):
    from aiohttp import web
    from .delivery import check_shared
    from .service import peer_api
    try:
        data = await request.json()
        if not isinstance(data, dict) or set(data) != {'server_url'}:
            raise ValueError('Expected server_url')
        return web.json_response(await asyncio.to_thread(check_shared, peer_api(data['server_url'], 5)))
    except (OSError, ValueError, RuntimeError) as exc:
        return web.json_response({'error': str(exc)}, status=400)


async def media(request):
    from aiohttp import web
    from .shared_store import read_receipt
    try:
        receipt = await asyncio.to_thread(read_receipt, request.match_info['request_id'])
        if receipt is None:
            raise FileNotFoundError('공유 결과 파일이 없습니다.')
        if not Path(receipt['path']).is_file():
            raise FileNotFoundError('결합된 최종 영상에서 확인하세요.')
        return web.FileResponse(receipt['path'], headers={'Cache-Control': 'no-store'})
    except (OSError, ValueError) as exc:
        return web.json_response({'error': str(exc)}, status=404)


def register_routes():
    try:
        from server import PromptServer
    except ImportError:
        return
    if getattr(PromptServer, "instance", None) is not None:
        routes = PromptServer.instance.routes
        routes.post(ROUTE)(discard)
        routes.post('/remote-vae/shared/check')(shared_check)
        routes.post('/remote-vae/preflight')(preflight)
        routes.get('/remote-vae/media/{request_id}')(media)

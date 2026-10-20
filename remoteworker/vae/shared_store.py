"""Durable shared latent handoff; delete input only after caller acknowledges video."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import uuid

from .codec import cpu_value, restore, validate_latent


def shared_root():
    location = os.environ.get('REMOTEWORKER_SHARED_VAE_DIR')
    if not location:
        cluster = os.environ.get('CLUSTER_ROOT')
        location = str(Path(cluster) / 'shared' / 'remote-vae') if cluster else None
    if not location or not Path(location).is_absolute():
        raise ValueError('REMOTEWORKER_SHARED_VAE_DIR에 두 서버가 공유하는 절대 경로를 설정하세요.')
    root = Path(location).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def request_path(area, request_id, name):
    if not isinstance(request_id, str) or not re.fullmatch(r'[a-f0-9]{32}', request_id):
        raise ValueError('Invalid shared VAE request ID')
    root = shared_root()
    path = root / area / request_id / name
    if path.resolve() != path:
        raise ValueError('Shared VAE path must not contain symlinks')
    return path


@contextmanager
def request_lock(request_id):
    # The lock covers decode, publication and acknowledgment, including retries.
    import fcntl
    path = request_path('temp', request_id, 'request.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.stem + '-' + uuid.uuid4().hex + path.suffix)
    try:
        writer(temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda temporary: temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8'))


def save_latent(samples, audio=None):
    import torch
    value = dict(validate_latent(samples))
    if audio is not None:
        value['remote_vae_audio'] = audio
    value = cpu_value(value)
    request_id = uuid.uuid4().hex
    path = request_path('temp', request_id, 'latent.pt')
    atomic_write(path, lambda temporary: torch.save(value, temporary))
    return request_id


def load_latent(request_id):
    import torch
    path = request_path('temp', request_id, 'latent.pt')
    if not path.is_file():
        raise ValueError(f'공유 latent가 없습니다: {request_id}. 완료 후 삭제되었거나 저장에 실패했습니다.')
    return validate_latent(restore(torch.load(path, map_location='cpu', weights_only=True)))


def media_path(request_id, options):
    return request_path('outputs', request_id, 'audio.wav' if options.get('kind') == 'audio' else 'video.mp4')


def read_receipt(request_id, options=None):
    receipt_path = request_path('outputs', request_id, 'receipt.json')
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    video = media_path(request_id, receipt.get('options', {}))
    consumed = False
    marker = receipt_path.with_name('consumed.json')
    if not video.exists() and marker.is_file():
        final_id = json.loads(marker.read_text(encoding='utf-8')).get('final_id')
        # One hop only: a final receipt cannot itself be a consumed source.
        final_path = request_path('outputs', final_id, 'receipt.json')
        if not final_path.with_name('consumed.json').exists():
            final = read_receipt(final_id)
            consumed = bool(final and final.get('final') is True and request_id in final.get('sources', []))
    if (receipt.get('request_id') != request_id or receipt.get('path') != str(video)
            or (not consumed and (not video.is_file() or video.stat().st_size <= 0
                or receipt.get('bytes') != video.stat().st_size))):
        raise ValueError('공유 영상 결과 경로 또는 파일 크기를 확인할 수 없습니다. latent를 보존합니다.')
    if options is not None and receipt.get('options') != options:
        raise ValueError('이미 완료된 작업의 설정이 다릅니다. 새 latent 저장 작업을 만드세요.')
    return receipt


def acknowledge(request_id, reported, options):
    with request_lock(request_id):
        receipt = read_receipt(request_id, options)
        if receipt is None or receipt != reported:
            raise ValueError('처리 서버와 호출 서버의 결과가 일치하지 않습니다. 공유 폴더를 확인하세요. latent를 보존합니다.')
        request_path('temp', request_id, 'latent.pt').unlink(missing_ok=True)
        # Keep the small lock file: removing a locked inode races waiting retries.
        return receipt['path']


def publish(request_id, options, encode):
    """Commit one video with an injected encoder; replay an existing commit."""
    with request_lock(request_id):
        receipt = read_receipt(request_id, options)
        if receipt is None:
            samples = load_latent(request_id)
            path = media_path(request_id, options)
            metadata = {}

            def write(temporary):
                metadata.update(encode(samples, temporary, options))

            atomic_write(path, write)
            receipt = dict(metadata, request_id=request_id, path=str(path),
                           bytes=path.stat().st_size, options=options)
            if receipt['bytes'] == 0:
                raise ValueError('Shared video encoder produced an empty file')
            write_json(request_path('outputs', request_id, 'receipt.json'), receipt)
        return receipt

"""Unified decode delivery: shared-storage handshake and typed result routing."""
from __future__ import annotations

import secrets
import uuid
from . import shared_store as store


def check_shared(api):
    """Verify both mounts by exchanging a request-scoped random challenge."""
    request_id, token = uuid.uuid4().hex, secrets.token_hex(24)
    probe = store.request_path('temp', request_id, 'probe.json')
    ack = probe.with_name('probe-ack.json')
    try:
        store.write_json(probe, {'token': token})
        result = api._json('/remote-vae/shared/check', payload={'request_id': request_id, 'token': token})
        import json
        if result.get('token') != token or json.loads(ack.read_text()).get('token') != token:
            raise ValueError('공유 폴더 확인 응답이 일치하지 않습니다.')
        return {'root': str(store.shared_root()), 'ready': True}
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError('공유 폴더 연결 실패: 양쪽 서버의 공유 경로와 읽기·쓰기 권한을 확인하세요. ' + str(exc)) from exc
    finally:
        for file in (probe, ack):
            file.unlink(missing_ok=True)
        if probe.parent.exists() and not any(probe.parent.iterdir()):
            probe.parent.rmdir()


def receive_probe(request_id, token):
    import json
    if not isinstance(token, str) or len(token) != 48:
        raise ValueError('Invalid shared storage challenge')
    path = store.request_path('temp', request_id, 'probe.json')
    if json.loads(path.read_text()).get('token') != token:
        raise ValueError('처리 서버에서 호출 서버의 공유 파일을 읽을 수 없습니다.')
    store.write_json(path.with_name('probe-ack.json'), {'token': token})
    return {'token': token}

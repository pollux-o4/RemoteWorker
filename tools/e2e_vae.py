"""Real VAE E2E: lossless tensor roundtrip or shared-file video lifecycle.

Shared mode uses native zero H3 latents; --large produces 175 x 512 x 512 RGB
frames (525 MiB). Both runtimes must mount REMOTEWORKER_SHARED_VAE_DIR identically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remoteworker.core.artifacts import successful_outputs
from remoteworker.core.execution import completed_artifacts, completed_prompt
from remoteworker.core.transport import HttpComfyApi
from remoteworker.vae import shared_store as store
from remoteworker.vae.codec import loads
from remoteworker.vae.schema import DECODE_DEFAULTS, VIDEO_FPS, VIDEO_CRF
from remoteworker.vae.service import RESULT


def node(class_type, **inputs):
    return {'class_type': class_type, 'inputs': inputs}


def run(api, graph):
    with completed_prompt(api, graph) as (prompt_id, history):
        return successful_outputs(history, prompt_id)


def verify_tensor(caller, peer_url, vae, kind, tiled, output_mode="tensor"):
    import torch
    peer = HttpComfyApi(peer_url, 1800)
    latent = node('EmptyMiniMaxH3LatentAV', width=32, height=32, length=5) if kind == 'audio' else node(
        'EmptySD3LatentImage', width=64, height=64, batch_size=1)
    options = DECODE_DEFAULTS | {'tiled': tiled}
    values, timings = [], []
    for remote, api in ((False, peer), (True, caller)):
        request_id = uuid.uuid4().hex
        graph = {'latent': latent}
        if remote:
            graph['warmup'] = node('RemoteVAEWarmup', vae_name=vae, kind=kind,
                server_url=peer_url, timeout_seconds=180, background=True)
            graph['decode'] = node('RemoteVAEDecodeAudio' if kind == 'audio' else 'RemoteVAEDecode',
                samples=['latent', 0], vae_name=vae, server_url=peer_url, timeout_seconds=180,
                warmup=['warmup', 0], memory_policy='release_after_result', output_mode=output_mode,
                **(options if kind == 'image' else {}))
        else:
            graph['vae'] = node('VAELoader', vae_name=vae)
            graph['decode'] = node('VAEDecodeAudio' if kind == 'audio' else 'VAEDecodeTiled' if tiled else 'VAEDecode',
                samples=['latent', 0], vae=['vae', 0],
                **({k: v for k, v in options.items() if k != 'tiled'} if kind == 'image' and tiled else {}))
        graph['save'] = node('SaveRemoteVAEOutput', value=['decode', 0], filename_prefix=f'remote-conditioning/{request_id}/decoded')
        started = time.monotonic()
        try:
            with completed_artifacts(api, graph, request_id, (RESULT,)) as blobs:
                values.append(loads(blobs['result']))
        finally:
            api.discard_vae_outputs(request_id)
        timings.append(round(time.monotonic()-started, 3))
    tensors = [value['waveform'] for value in values] if kind == 'audio' else values
    torch.testing.assert_close(*tensors, rtol=0, atol=0)
    if kind == 'audio':
        assert values[0]['sample_rate'] == values[1]['sample_rate'] > 0
    return dict(kind=kind, tiled=tiled, native_seconds=timings[0], remote_seconds=timings[1],
                shape=list(tensors[1].shape), bit_exact=True)


def verify_shared(caller, peer_url, vae, audio_vae, large, tiled):
    import av
    size, frames = (512, 175) if large else (64, 5)
    options = dict(vae_name=vae, audio_vae_name=audio_vae, fps=VIDEO_FPS, crf=VIDEO_CRF,
                   tiled=tiled, server_url=peer_url, timeout_seconds=1800, memory_policy='release_after_result')
    started = time.monotonic()
    latent_graph = {'latent': node('EmptyMiniMaxH3LatentAV', width=size, height=size, length=frames),
                    'save': node('SaveSharedVAELatent', samples=['latent', 0])}
    outputs = run(caller, {**latent_graph, 'video': node('RemoteVAEVideo', latent=['save', 0], **options)})
    request_id, path = outputs['save']['text'][0], Path(outputs['video']['text'][0])
    assert str(path) == store.read_receipt(request_id)['path']
    assert not store.request_path('temp', request_id, 'latent.pt').exists()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height, sum(1 for _ in container.decode(stream))) == (size, size, frames)
    with av.open(str(path)) as container:
        assert bool(container.streams.audio) == bool(audio_vae)
        audio_samples = sum(frame.samples for frame in container.decode(audio=0)) if audio_vae else 0
        assert not audio_vae or audio_samples > 0
    before = path.stat().st_mtime_ns
    assert run(caller, {'video': node('RemoteVAEVideo', request_id=request_id, **options)})['video']['text'][0] == str(path)
    assert path.stat().st_mtime_ns == before, 'Retry encoded the video again'
    if audio_vae:
        # Valid model, wrong media family: real peer failure must preserve input.
        failed_id = run(caller, latent_graph)['save']['text'][0]
        pid = caller.submit({'video': node('RemoteVAEVideo', request_id=failed_id,
            **(options | {'vae_name': audio_vae, 'audio_vae_name': ''}))})
        assert caller.wait_for_history(pid)[pid]['status']['status_str'] == 'error'
        assert store.request_path('temp', failed_id, 'latent.pt').is_file()
        recovered = run(caller, {'video': node('RemoteVAEVideo', request_id=failed_id, **options)})
        assert Path(recovered['video']['text'][0]).is_file()
        assert not store.request_path('temp', failed_id, 'latent.pt').exists()
    return dict(request_id=request_id, video=str(path), frames=frames, width=size, height=size,
                decoded_mib=frames*size*size*12/1024**2, video_bytes=path.stat().st_size,
                audio_samples=audio_samples, retry_without_reencoding=True,
                peer_failure_recovered=bool(audio_vae), latent_deleted=True, seconds=round(time.monotonic()-started, 2))


def verify_unified(caller, peer_url, vae, audio_vae, large, kind, tiled):
    if large or audio_vae:
        raise ValueError("통합 텐서 검증은 소형 단일 VAE 대상입니다. 대용량·결합 영상 저장 검증은 --mode shared를 사용하세요.")
    return verify_tensor(caller, peer_url, vae, kind, tiled, output_mode="shared")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('caller', 'peer', 'vae'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=('tensor', 'shared', 'unified'), default='shared')
    parser.add_argument('--kind', choices=('audio', 'image'), default='audio')
    parser.add_argument('--audio-vae', default='')
    parser.add_argument('--tiled', action='store_true')
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--report')
    args = parser.parse_args()
    caller = HttpComfyApi(args.caller, 1800)
    report = verify_unified(caller, args.peer, args.vae, args.audio_vae, args.large, args.kind, args.tiled) if args.mode == 'unified' else verify_shared(caller, args.peer, args.vae, args.audio_vae, args.large, args.tiled) if args.mode == 'shared' else verify_tensor(
        caller, args.peer, args.vae, args.kind, args.tiled)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text+'\n', encoding='utf-8')
    print(text)

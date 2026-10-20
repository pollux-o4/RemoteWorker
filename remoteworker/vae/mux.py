"""Shared MP4/WAV finalization: stream-copy video, atomic publication, owned cleanup."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess

from . import shared_store as store


def run(command, interrupt=None):
    if interrupt:
        interrupt()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            while True:
                try:
                    out, err = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if interrupt:
                        interrupt()
            if process.returncode:
                raise RuntimeError('미디어 처리 실패: ' + err.decode(errors='replace')[-1200:])
            return out
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


def probe(path, interrupt=None):
    return json.loads(run(['ffprobe', '-v', 'error', '-show_entries',
        'stream=codec_type,codec_name,width,height,sample_rate,channels,duration:format=duration',
        '-of', 'json', str(path)], interrupt))


def source_id(path, filename):
    path = Path(path)
    identity = path.parent.name
    expected = store.request_path('outputs', identity, filename)
    if path != expected:
        raise ValueError('공유 VAE가 반환한 파일 경로를 연결하세요.')
    return identity


def encode(video, audio, target, interrupt=None):
    # Preserve encoded video. PCM WAV becomes browser-compatible AAC; pad short
    # audio so -shortest follows video rather than truncating picture frames.
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video)]
    if audio:
        command += ['-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-af', 'apad', '-shortest']
    else:
        command += ['-map', '0:v:0', '-map', '0:a?', '-c', 'copy']
    run(command + ['-movflags', '+faststart', '-y', str(target)], interrupt)


def validate(path, audio_required, interrupt=None):
    info = probe(path, interrupt)
    streams = info.get('streams', [])
    videos = [s for s in streams if s.get('codec_type') == 'video']
    audios = [s for s in streams if s.get('codec_type') == 'audio']
    if len(videos) != 1 or float(info.get('format', {}).get('duration', 0)) <= 0:
        raise ValueError('완성 영상의 비디오 트랙·길이를 확인할 수 없습니다.')
    if audio_required and (not audios or any(int(s.get('sample_rate', 0)) <= 0 for s in audios)):
        raise ValueError('완성 영상에 오디오 트랙이 없습니다. 중간 파일을 보존합니다.')
    return dict(audio=bool(audios), duration=float(info['format']['duration']),
                width=videos[0].get('width'), height=videos[0].get('height'))


def combine(video_path, audio_path='', *, cleanup=True, interrupt=None):
    video_id = source_id(video_path, 'video.mp4')
    audio_id = source_id(audio_path, 'audio.wav') if audio_path else None
    sources = [video_id] + ([audio_id] if audio_id else [])
    identity = hashlib.sha256(('mux-v1:' + ':'.join(sources)).encode()).hexdigest()[:32]
    options = {'kind': 'image', 'sources': sources, 'mux': 'h264-copy-aac-v1'}
    with ExitStack() as locks:
        # Ordered locks prevent competing finalizers from consuming the same input.
        for key in sorted(set([identity, *sources])):
            locks.enter_context(store.request_lock(key))
        receipt = store.read_receipt(identity, options)
        if receipt is None:
            for key in sources:
                if store.read_receipt(key) is None:
                    raise ValueError('디코딩 결과가 없습니다. 원격 VAE를 다시 실행하세요.')
            video = Path(video_path)
            audio = Path(audio_path) if audio_path else None
            video_info = probe(video, interrupt)
            video_duration = float(video_info.get('format', {}).get('duration', 0))
            if audio:
                audio_info = probe(audio, interrupt)
                duration = float(audio_info.get('format', {}).get('duration', 0))
                if not any(s.get('codec_type') == 'audio' for s in audio_info.get('streams', [])):
                    raise ValueError('연결된 음성 파일에 오디오 트랙이 없습니다.')
                if abs(video_duration - duration) > 0.25:
                    raise ValueError('영상과 음성 길이가 다릅니다. 중간 파일은 보존했습니다. 연결한 작업을 확인하세요.')
            path = store.media_path(identity, options)
            metadata = {}

            def write(temporary):
                encode(video, audio, temporary, interrupt)
                metadata.update(validate(temporary, bool(audio), interrupt))
                if abs(metadata['duration'] - video_duration) > 0.25:
                    raise ValueError('결합 후 영상 길이가 달라졌습니다. 중간 파일을 보존합니다.')

            store.atomic_write(path, write)
            receipt = dict(metadata, request_id=identity, path=str(path), bytes=path.stat().st_size,
                           options=options, sources=sources, final=True)
            store.write_json(path.with_name('receipt.json'), receipt)
        if cleanup:
            # Publish first. Retrying this node reads the final receipt before
            # touching inputs, including after interrupted cleanup.
            for key, filename in zip(sources, ['video.mp4', 'audio.wav']):
                source = store.request_path('outputs', key, filename)
                store.write_json(source.with_name('consumed.json'), {'final_id': identity})
                source.unlink(missing_ok=True)
                store.request_path('temp', key, 'latent.pt').unlink(missing_ok=True)
        return receipt

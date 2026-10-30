"""Shared MP4/WAV finalization: stream-copy video, atomic publication, owned cleanup."""

from __future__ import annotations

import json
import subprocess

from .. import _native
from .._native import vae_mux_source_id as source_id


def run(command, interrupt=None):
    if interrupt:
        interrupt()
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        try:
            while True:
                try:
                    out, err = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    if interrupt:
                        interrupt()
            if process.returncode:
                raise RuntimeError(
                    "미디어 처리 실패: " + err.decode(errors="replace")[-1200:]
                )
            return out
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


def probe(path, interrupt=None):
    return json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,sample_rate,channels,duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            interrupt,
        )
    )


def encode(video, audio, target, interrupt=None):
    run(_native.vae_mux_command(video, audio, target), interrupt)


def validate(path, audio_required, interrupt=None):
    return _native.vae_mux_validate(probe(path, interrupt), audio_required)


def combine(video_path, audio_path="", *, cleanup=True, interrupt=None):
    return _native.vae_mux_combine(
        str(video_path), str(audio_path), cleanup, interrupt, encode, probe
    )


__all__ = ["run", "probe", "source_id", "encode", "validate", "combine"]

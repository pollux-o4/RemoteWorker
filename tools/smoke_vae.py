"""Compare an isolated caller→peer audio decode against the native decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remoteworker.core.execution import completed_artifacts
from remoteworker.core.transport import HttpComfyApi
from remoteworker.vae.codec import loads
from remoteworker.vae.service import RESULT


def verify(caller_url, peer_url, vae_name, kind="audio", tiled=False, background=True):
    import torch
    caller, peer = HttpComfyApi(caller_url, 180), HttpComfyApi(peer_url, 180)
    common = {"latent": {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {
        "width": 32, "height": 32, "length": 5}}}
    if kind == "image":
        common = {"latent": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": 64, "height": 64, "batch_size": 1}}}
    options = dict(tiled=tiled, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
    outputs, timings, request_ids = [], [], []
    for remote in (False, True):
        request_id = "vae-qa-" + uuid.uuid4().hex
        request_ids.append(request_id)
        graph = dict(common)
        if remote:
            graph.update({
                "warmup": {"class_type": "RemoteVAEWarmup", "inputs": {
                    "vae_name": vae_name, "kind": kind, "server_url": peer_url, "timeout_seconds": 180, "background": background}},
                "decode": {"class_type": "RemoteVAEDecodeAudio" if kind == "audio" else "RemoteVAEDecode", "inputs": {
                    "samples": ["latent", 0], "vae_name": vae_name, "server_url": peer_url,
                    "timeout_seconds": 180, "memory_policy": "release_after_result", "warmup": ["warmup", 0],
                    **(options if kind == "image" else {})}},
            })
        else:
            graph.update({
                "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
                "decode": {"class_type": "VAEDecodeAudio" if kind == "audio" else ("VAEDecodeTiled" if tiled else "VAEDecode"), "inputs": {
                    "samples": ["latent", 0], "vae": ["vae", 0],
                    **({k: v for k, v in options.items() if k != "tiled"} if kind == "image" and tiled else {})}},
            })
        graph["save"] = {"class_type": "SaveRemoteVAEOutput", "inputs": {
            "value": ["decode", 0], "filename_prefix": f"remote-conditioning/{request_id}/decoded"}}
        started = time.monotonic()
        with completed_artifacts(caller if remote else peer, graph, request_id, (RESULT,)) as blobs:
            outputs.append(loads(blobs["result"]))
        timings.append(round(time.monotonic() - started, 3))
    values = [item["waveform"] for item in outputs] if kind == "audio" else outputs
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    if kind == "audio":
        assert outputs[0]["sample_rate"] == outputs[1]["sample_rate"] == 32000
    return {"kind": kind, "tiled": tiled, "background_warmup": background, "native_seconds": timings[0],
        "remote_with_warmup_seconds": timings[1], "shape": list(values[1].shape),
        "bit_exact": True, "request_ids": request_ids}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caller", required=True)
    parser.add_argument("--peer", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--kind", choices=("audio", "image"), default="audio")
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.caller, args.peer, args.vae, args.kind, args.tiled), indent=2))

"""Verify a running, isolated ComfyUI using small CPU fixtures; no model inference."""
from __future__ import annotations

import argparse
import io
import sys
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remoteworker.core.artifacts import RemoteArtifactSpec
from remoteworker.core.execution import RequestInputs, completed_artifacts
from remoteworker.core.transport import HttpComfyApi


def verify(url: str, directory: Path) -> None:
    request_id = f"qa-{uuid.uuid4().hex}"
    api = HttpComfyApi(url, 30)
    required = (
        "SaveConditioning", "LoadConditioning", "SaveRemoteConditioning", "SaveRemoteLatent",
        "SaveRemoteText", "RemoteConditioning", "RemoteMiniMaxH3ReferenceEncode",
        "RemoteMiniMaxH3ImageToVideo", "RemotePromptEnhancer", "RemoteVAESelector",
        "RemoteQwenImage21Encode", "RemoteQwenImage21",
    )
    for node in required:
        api.require_node(node)
    fixture = [[torch.arange(6, dtype=torch.float32).reshape(1, 2, 3), {"pooled_output": torch.ones(1, 3)}]]
    source = directory / "input" / f"{request_id}.cond"
    source.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, source)
    output = directory / "output" / "remote-conditioning" / request_id
    workflow = {
        "load": {"class_type": "LoadConditioning", "inputs": {"filename": source.name}},
        "save": {"class_type": "SaveRemoteConditioning", "inputs": {
            "conditioning": ["load", 0], "filename_prefix": f"remote-conditioning/{request_id}/conditioning"}},
        "text": {"class_type": "SaveRemoteText", "inputs": {
            "text": "RemoteWorker QA", "filename_prefix": f"remote-conditioning/{request_id}/prompt"}},
    }
    specs = (
        RemoteArtifactSpec("conditioning", "save", "conditioning", ".cond"),
        RemoteArtifactSpec("prompt", "text", "text", ".txt"),
    )
    try:
        # Exercise the real upload/cleanup boundary as well as artifact recovery.
        from PIL import Image
        image = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        with RequestInputs(api, request_id) as inputs:
            inputs.upload_image(index=1, png=image.getvalue())
            with completed_artifacts(api, workflow, request_id, specs) as blobs:
                result = torch.load(io.BytesIO(blobs["conditioning"]), map_location="cpu", weights_only=False)
                assert torch.equal(result[0][0], fixture[0][0])
                assert torch.equal(result[0][1]["pooled_output"], fixture[0][1]["pooled_output"])
                assert blobs["prompt"] == b"RemoteWorker QA"
        assert not (directory / "temp" / "remote-conditioning" / request_id).exists()
        print(f"PASS {url}: 12 nodes; upload → submit → history → scoped artifacts → tensor equality → cleanup")
    finally:
        source.unlink(missing_ok=True)
        if output.is_dir():
            for artifact in output.iterdir():
                artifact.unlink()
            output.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--base-directory", type=Path, required=True, help="Isolated server base directory")
    args = parser.parse_args()
    verify(args.url, args.base_directory.resolve())

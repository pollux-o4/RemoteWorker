from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import torch

import remoteworker.qwen.service as qwen


class FakeFrame:
    shape = (1, 64, 96, 3)

    def __getitem__(self, index):
        return self


class FakeApi:
    def __init__(self):
        self.events = []
        self.workflow = None

    def require_node(self, name):
        self.events.append(("capability", name))
        if name == qwen.QWEN_WORKER:
            return {"input": {"required": {key: [] for key in ("clip", "prompt", "negative_prompt", "resolution")}, "optional": {key: [] for key in ("vae", "negative_vision", "image_1")}}, "output": ["CONDITIONING", "CONDITIONING", "LATENT"]}
        if name == "VAELoader":
            return {"input": {"required": {"vae_name": [["qwen-vae.safetensors"]]}}}
        if name == "VRAMCleanup":
            return {"input": {"required": {"offload_model": [], "offload_cache": [], "reset_execution_cache": []}, "optional": {"anything": ["*"]}}, "output": ["*"]}
        raise AssertionError(name)

    def clip_loader_options(self):
        return {"clip_name": ["qwen-text-encoder.safetensors"], "clip_type": ["qwen_image"], "device": ["default"]}

    def upload_image(self, **kwargs):
        self.events.append(("upload", kwargs["index"]))
        return f"remote-conditioning/request-1/image-{kwargs['index']:04d}.png [temp]"

    def submit(self, workflow):
        self.workflow = workflow
        return "prompt-1"

    def wait_for_history(self, _prompt_id):
        base = "remote-conditioning/request-1"
        return {"prompt-1": {"status": {"status_str": "success"}, "outputs": {
            "save_positive": {"conditioning": [f"{base}/positive_00001_.cond"]},
            "save_negative": {"conditioning": [f"{base}/negative_00001_.cond"]},
            "save_latent": {"latents": [f"{base}/latent_00001_.latent"]},
        }}}

    def download_artifact(self, path):
        return path.encode()

    def discard_inputs(self, request_id):
        self.events.append(("discard", request_id))


class RemoteQwenImage21Tests(unittest.TestCase):
    def test_workflow_preserves_native_three_output_contract_and_orders_cleanup(self):
        workflow = qwen.build_workflow(
            clip=qwen.ClipSelection("qwen-text-encoder.safetensors", "qwen_image", "default"),
            vae_name="qwen-vae.safetensors", prompt="edit", negative_prompt="", resolution=1024,
            negative_vision=False, image_refs=["remote-conditioning/request-1/image-0001.png [temp]"], request_id="request-1",
        )
        self.assertEqual(qwen.QWEN_WORKER, workflow["encode"]["class_type"])
        self.assertFalse(workflow["encode"]["inputs"]["negative_vision"])
        self.assertEqual(["image_1_loader", 0], workflow["encode"]["inputs"]["image_1"])
        self.assertEqual(qwen.SAVE_REMOTE_CONDITIONING, workflow["save_positive"]["class_type"])
        self.assertEqual(["save_positive", 0], workflow["save_negative"]["inputs"]["dependency"])
        self.assertEqual(["save_negative", 0], workflow["save_latent"]["inputs"]["dependency"])
        self.assertEqual(["save_latent", 0], workflow["cleanup"]["inputs"]["anything"])

    def test_execute_validates_then_uploads_and_recovers_three_scoped_artifacts(self):
        api = FakeApi()
        with patch.object(qwen, "encode_image_frame_png", return_value=b"png"):
            result = qwen.execute(
                clip=qwen.ClipSelection("qwen-text-encoder.safetensors", "qwen_image", "default"), vae_name="qwen-vae.safetensors",
                prompt="edit", negative_prompt="", resolution=1024, negative_vision=False, frames=[FakeFrame()], request_id="request-1", api=api,
            )
        self.assertEqual(tuple(path.encode() for path in (
            "remote-conditioning/request-1/positive_00001_.cond",
            "remote-conditioning/request-1/negative_00001_.cond",
            "remote-conditioning/request-1/latent_00001_.latent",
        )), result)
        self.assertEqual(("capability", qwen.QWEN_WORKER), api.events[0])
        self.assertEqual(("upload", 1), [event for event in api.events if event[0] == "upload"][0])
        self.assertEqual(("discard", "request-1"), api.events[-1])

    def test_master_node_round_trips_cpu_artifacts_and_caches_same_inputs(self):
        streams = []
        for value in (["positive"], ["negative"], {"samples": torch.zeros([1, 64, 2, 2])}):
            stream = io.BytesIO()
            torch.save(value, stream)
            streams.append(stream.getvalue())
        node = qwen.RemoteQwenImage21Service()
        with patch.object(qwen, "HttpComfyApi", return_value=object()), patch.object(qwen, "execute", return_value=tuple(streams)) as execute:
            first = node.run("edit", "", "http://127.0.0.1:8188", 600, "qwen-text-encoder.safetensors", "qwen-vae.safetensors", 1024, False)
            second = node.run("edit", "", "http://127.0.0.1:8188", 600, "qwen-text-encoder.safetensors", "qwen-vae.safetensors", 1024, False)
        self.assertEqual(["positive"], first[0])
        self.assertEqual(["negative"], first[1])
        self.assertIn("samples", first[2])
        self.assertEqual(first[0], second[0])
        self.assertEqual(1, execute.call_count)

    @unittest.skipUnless(qwen.comfy_io is not None, "Native schema needs ComfyUI on PYTHONPATH")
    def test_native_schema_preserves_outputs_autogrow_and_cache_fingerprint(self):
        import math
        schema = qwen.RemoteQwenImage21Node.define_schema()
        self.assertEqual("RemoteQwenImage21", schema.node_id)
        self.assertEqual(["CONDITIONING", "CONDITIONING", "LATENT"], [output.io_type for output in schema.outputs])
        images = next(item for item in schema.inputs if item.id == "images")
        self.assertEqual([f"image_{index}" for index in range(1, 17)], images.template.names)
        self.assertEqual(0, images.template.min)
        self.assertEqual(0, qwen.RemoteQwenImage21Node.fingerprint_inputs(cache_policy="reuse_same_inputs"))
        self.assertTrue(math.isnan(qwen.RemoteQwenImage21Node.fingerprint_inputs(cache_policy="always_refresh")))

    def test_service_uses_its_single_inline_connection_configuration(self):
        streams = []
        for value in (["positive"], ["negative"], {"samples": torch.zeros([1, 64, 2, 2])}):
            stream = io.BytesIO()
            torch.save(value, stream)
            streams.append(stream.getvalue())
        with patch.object(qwen, "HttpComfyApi", return_value=object()), patch.object(qwen, "execute", return_value=tuple(streams)) as execute:
            qwen.RemoteQwenImage21Service().run("edit", "", "http://127.0.0.1:8188", 30, "qwen-text-encoder.safetensors", "qwen-vae.safetensors", 1024, False, cache_policy="always_refresh")
        self.assertEqual("qwen-text-encoder.safetensors", execute.call_args.kwargs["clip"].clip_name)
        self.assertEqual("qwen-vae.safetensors", execute.call_args.kwargs["vae_name"])

    def test_peer_refreshed_model_values_bypass_only_the_static_schema_allowlist(self):
        peer_clip = "remote-models\\example-vision-encoder.safetensors"

        self.assertTrue(qwen._validate_dynamic_peer_models(peer_clip, "qwen-vae.safetensors"))
        self.assertIn("must be a non-empty string", qwen._validate_dynamic_peer_models("", "qwen-vae.safetensors"))

    def test_images_must_be_consecutive_and_cannot_exceed_the_official_slot_cap(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            qwen._ordered_qwen_image_frames(image_1=FakeFrame(), image_3=FakeFrame())
        self.assertEqual([], qwen._ordered_qwen_image_frames())


if __name__ == "__main__":
    unittest.main()

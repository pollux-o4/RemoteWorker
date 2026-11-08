from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from remoteworker.vae import codec, service, worker

OPTIONS = dict(tiled=False, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)


class FakeApi:
    def __init__(self, error=False):
        self.events = []
        self.graph = None
        self.error = error

    def require_node(self, name):
        self.events.append(name)
        return {"input": {"required": {"vae_name": [["test.vae"]]}}}

    def upload_file(self, **kwargs):
        self.events.append("upload")
        self.request_id = kwargs["request_id"]
        self.source = codec.loads(kwargs["data"])
        return f"remote-conditioning/{self.request_id}/samples.pt [temp]"

    def submit(self, graph):
        self.graph = graph
        return "job"

    def wait_for_history(self, prompt_id):
        if self.error:
            raise TimeoutError("peer timed out")
        return {
            "job": {
                "status": {"status_str": "success"},
                "outputs": {"save": {"vae_output": [f"remote-conditioning/{self.request_id}/decoded.pt"]}},
            }
        }

    def download_artifact(self, path):
        return codec.dumps(torch.zeros(1, 4, 4, 3))

    def cancel_prompt(self, prompt_id):
        self.events.append("cancel")

    def discard_vae_outputs(self, request_id):
        self.events.append("discard-output")
        return True

    def discard_inputs(self, request_id):
        self.events.append("discard")


class RemoteVAETests(unittest.TestCase):
    def test_lossless_cpu_detached_roundtrip(self):
        source = torch.rand(1, 24, 2, 2, 2, requires_grad=True)
        result = codec.loads(codec.dumps({"samples": source, "sample_rate": 32000}))
        self.assertTrue(torch.equal(source, result["samples"]))
        self.assertFalse(result["samples"].requires_grad)
        self.assertEqual("cpu", result["samples"].device.type)
        self.assertEqual(32000, result["sample_rate"])

    def test_rejects_arbitrary_metadata_and_excessive_envelopes(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            codec.dumps({"samples": object()})
        with patch.object(codec, "MAX_BYTES", 4):
            with self.assertRaises(ValueError):
                codec.dumps(torch.ones(10))
        with self.assertRaises(ValueError):
            codec.loads(b"")

    def test_invalid_latent_and_result_rejected(self):
        for value in ({}, {"samples": "wrong"}):
            with self.assertRaises(ValueError):
                codec.validate_latent(value)
        with self.assertRaises(ValueError):
            codec.validate_result(torch.zeros(1, 4), "image")
        with self.assertRaises(ValueError):
            codec.validate_result({"waveform": torch.zeros(1, 2, 100), "sample_rate": 0}, "audio")

    def test_decode_uploads_cpu_latent_and_cleans_request(self):
        api = FakeApi()
        result = service.execute_decode(
            api,
            samples={"samples": torch.zeros(1, 4, 8, 8)},
            vae_name="test.vae",
            kind="image",
            options=OPTIONS,
            memory_policy="keep",
        )
        self.assertEqual((1, 4, 4, 3), result.shape)
        self.assertEqual("discard", api.events[-1])
        self.assertNotIn("cleanup", api.graph)
        self.assertEqual("VAELoader", api.graph["vae"]["class_type"])

    def test_decode_failure_cancels_only_its_prompt_and_discards_inputs(self):
        api = FakeApi(error=True)
        with self.assertRaises(TimeoutError):
            service.execute_decode(
                api,
                samples={"samples": torch.zeros(1, 4, 8, 8)},
                vae_name="test.vae",
                kind="image",
                options=OPTIONS,
                memory_policy="keep",
            )
        self.assertEqual(["cancel", "discard-output", "discard"], api.events[-3:])

    def test_unavailable_vae_rejected_before_upload(self):
        api = FakeApi()
        with self.assertRaisesRegex(ValueError, "선택한 VAE가 없습니다"):
            service.execute_decode(
                api,
                samples={},
                vae_name="missing",
                kind="image",
                options=OPTIONS,
                memory_policy="keep",
            )
        self.assertNotIn("upload", api.events)

    def test_cleanup_order_is_after_output_serialization(self):
        graph = service.build_workflow("test", "file", "image", OPTIONS, "req", "release_after_result")
        self.assertEqual(["save", 0], graph["cleanup"]["inputs"]["anything"])
        self.assertFalse(graph["cleanup"]["inputs"]["reset_execution_cache"])

    def test_warmup_uses_minimal_zero_latent_for_each_media_family(self):
        for kind, channels, dims, shape in (
            ("audio", 32, 2, (1, 32, 2, 8)),
            ("image", 24, 3, (1, 24, 2, 2, 2)),
        ):
            vae = SimpleNamespace(latent_channels=channels, latent_dim=dims)
            with (
                self.subTest(kind=kind),
                patch.object(worker, "decode", return_value={}) as call,
            ):
                self.assertEqual(("ready",), worker.RemoteVAEWarmupWorker().run(vae, kind))
            self.assertTrue(torch.equal(call.call_args.args[1]["samples"], torch.zeros(shape)))

    def test_warmup_always_executes_and_has_ordering_input(self):
        import math

        self.assertTrue(math.isnan(service.RemoteVAEWarmup.IS_CHANGED()))
        self.assertTrue(math.isnan(worker.RemoteVAEWarmupWorker.IS_CHANGED()))
        self.assertIn("after", service.RemoteVAEWarmup.INPUT_TYPES()["optional"])
        self.assertIn("warmup", service.RemoteVAEDecode.INPUT_TYPES()["optional"])
        self.assertEqual(("AUDIO", "STRING"), service.RemoteVAEDecodeAudio.RETURN_TYPES)
        self.assertNotIn("tiled", service.RemoteVAEDecodeAudio.INPUT_TYPES()["required"])

    def test_worker_rejects_paths_outside_request_upload(self):
        for path in (
            "/etc/passwd",
            "remote-conditioning/../samples.pt [temp]",
            "samples.pt",
        ):
            with self.assertRaises(ValueError):
                worker.load_request_latent(path)

    def test_warmup_receipt_verified_and_keeps_models(self):
        api = FakeApi()
        api.wait_for_history = lambda _: {
            "job": {
                "status": {"status_str": "success"},
                "outputs": {"save": {"text": [api.graph["save"]["inputs"]["filename_prefix"] + ".txt"]}},
            }
        }
        api.download_artifact = lambda _: b"ready"
        import json

        ticket = service.execute_warmup(api, vae_name="test.vae", kind="audio", server_url="http://peer")
        self.assertEqual("test.vae", json.loads(ticket)["vae_name"])
        self.assertEqual("http://peer", json.loads(ticket)["server_url"])
        self.assertNotIn("cleanup", api.graph)

    def test_background_warmup_does_not_wait_and_preserves_passthrough(self):
        api = FakeApi()
        api.wait_for_history = lambda _: self.fail("background warmup must not wait")
        ticket = service.execute_warmup(
            api,
            vae_name="test.vae",
            kind="audio",
            server_url="http://peer",
            background=True,
        )
        import json

        self.assertEqual("job", json.loads(ticket)["prompt_id"])
        conditioning = [[torch.ones(1), {}]]
        with patch.object(service, "peer_api", return_value=api):
            result = service.RemoteVAEWarmup().run("test.vae", "audio", "http://peer", after=conditioning)
        self.assertIs(conditioning, result[1])

    def test_warmup_ticket_cannot_join_a_different_peer_or_model(self):
        from remoteworker.vae import warmup

        api = FakeApi()
        ticket = warmup.schedule(api, vae_name="test.vae", kind="audio", server_url="http://peer")
        for changes in (
            {"server_url": "http://other"},
            {"vae_name": "other"},
            {"kind": "image"},
        ):
            options = dict(vae_name="test.vae", kind="audio", server_url="http://peer") | changes
            with self.assertRaisesRegex(ValueError, "다릅니다"):
                warmup.finish(api, ticket, **options)

    def test_self_queue_calls_rejected_before_submission(self):
        import sys

        fake = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(port=18289)))
        with patch.dict(sys.modules, {"server": fake}):
            with self.assertRaisesRegex(ValueError, "다른"):
                service.peer_api("http://127.0.0.1:18289", 30)

    def test_large_video_rejected_before_decode_allocation(self):
        vae = SimpleNamespace(upscale_ratio=(lambda t: t * 4, 16, 16), output_channels=3)
        with self.assertRaisesRegex(ValueError, "budget"):
            worker.check_output_budget(vae, {"samples": torch.empty(1, 24, 50, 32, 32)}, "image")

    def test_output_cleanup_preserves_unrelated_files_and_rejects_symlinks(self):
        from remoteworker.vae.routes import discard_output

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "remote-conditioning" / "request-1"
            path.mkdir(parents=True)
            (path / "decoded_00001_.pt").write_bytes(b"tensor")
            (path / "keep.txt").write_text("keep")
            self.assertTrue(discard_output("request-1", root))
            self.assertEqual(["keep.txt"], [p.name for p in path.iterdir()])
            with self.assertRaises(ValueError):
                discard_output("../bad", root)
            alias = path.parent / "alias"
            alias.symlink_to(root)
            with self.assertRaises(ValueError):
                discard_output("alias", root)


if __name__ == "__main__":
    unittest.main()

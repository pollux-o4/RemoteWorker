"""Actual shared tensor worker/storage/consumer; replace only HTTP and native VAE decode."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from remoteworker.vae import codec, delivery, service, shared_tensor as transfer, shared_store as store


class Peer:
    fail_history = False
    fail_cleanup = False

    def require_node(self, name):
        return {"input": {"required": {"vae_name": [["test.vae"]]}}}

    def _json(self, path, *, payload):
        return delivery.receive_probe(**payload)

    def submit(self, graph):
        self.graph = graph
        return "shared-job"

    def wait_for_history(self, prompt_id):
        result = transfer.SharedVAETensorWorker().run(**self.graph["save"]["inputs"])
        if self.fail_history:
            raise TimeoutError("lost history")
        status = "error" if self.fail_cleanup else "success"
        return {prompt_id: {"status": {"status_str": status}, "outputs": {"save": result["ui"]}}}

    def cancel_prompt(self, prompt_id):
        self.cancelled = prompt_id


class SharedTensorContracts(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, REMOTEWORKER_SHARED_VAE_DIR=str(self.root)))
        self.peer = Peer()
        self.native = self.enterContext(patch("remoteworker.vae.worker.decode"))
        self.enterContext(patch.dict("sys.modules", nodes=SimpleNamespace(VAELoader=lambda: SimpleNamespace(load_vae=lambda name: (object(),)))))
        self.samples = {"samples": torch.ones(1, 16, 1, 8, 8)}
        self.options = transfer.options("test.vae", "image")

    def run_transfer(self, kind="image", request_id=""):
        return transfer.execute(self.peer, self.samples, "test.vae", kind, "http://peer", "keep", {}, request_id)

    def test_image_batch_transfers_exactly_without_final_media_or_http_budget(self):
        # A batch must remain a batch of images, never become an MP4.
        expected = torch.rand(3, 32, 48, 3)
        self.native.return_value = expected
        with patch.object(codec, "MAX_BYTES", 1):
            actual = self.run_transfer()
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(self.native.call_args.kwargs["transfer"])
        self.assertEqual(list(self.root.rglob("*.pt")), [])
        self.assertEqual(list(self.root.rglob("*.json")), [])
        self.assertEqual(list(self.root.glob("outputs/**/*")), [])

    def test_audio_returns_waveform_and_sample_rate_without_wav(self):
        expected = {"waveform": torch.rand(1, 2, 80), "sample_rate": 32000}
        self.native.return_value = expected
        actual = self.run_transfer("audio")
        self.assertTrue(torch.equal(actual["waveform"], expected["waveform"]))
        self.assertEqual(actual["sample_rate"], 32000)
        self.assertEqual(list(self.root.rglob("*.wav")), [])
        self.assertEqual(list(self.root.rglob("*.pt")), [])

    def test_lost_history_preserves_retry_and_reuses_decoded_tensor(self):
        self.native.return_value = torch.rand(1, 32, 48, 3)
        request_id = store.save_latent(self.samples)
        self.peer.fail_history = True
        with self.assertRaisesRegex(RuntimeError, request_id):
            self.run_transfer(request_id=request_id)
        self.assertTrue(store.media_path(request_id, self.options).is_file())
        self.assertTrue(store.request_path("temp", request_id, "latent.pt").is_file())
        self.peer.fail_history = False
        self.assertTrue(torch.equal(self.run_transfer(request_id=request_id), self.native.return_value))
        self.assertEqual(self.native.call_count, 1)
        self.assertEqual(list(self.root.rglob("*.pt")), [])

    def test_invalid_tensor_never_acknowledges_or_deletes_input(self):
        request_id = store.save_latent(self.samples)
        def bad(samples, path, settings):
            torch.save(torch.ones(3), path)
            return {}
        receipt = store.publish(request_id, self.options, bad)
        with self.assertRaisesRegex(ValueError, "invalid image"):
            transfer.consume(request_id, receipt, self.options)
        self.assertTrue(store.request_path("temp", request_id, "latent.pt").exists())
        self.assertTrue(store.media_path(request_id, self.options).exists())

    def test_changed_receipt_or_settings_never_consumes_input(self):
        self.native.return_value = torch.rand(1, 32, 48, 3)
        request_id = store.save_latent(self.samples)
        receipt = transfer.SharedVAETensorWorker().run(request_id, "test.vae", "image")["ui"]["shared_tensor"][0]
        for reported, settings in ((receipt | {"bytes": 1}, self.options),
                                    (receipt, self.options | {"tiled": True})):
            with self.assertRaises(ValueError):
                transfer.consume(request_id, reported, settings)
            self.assertTrue(store.request_path("temp", request_id, "latent.pt").exists())

    def test_cleanup_failure_preserves_shared_transfer_for_retry(self):
        self.native.return_value = torch.rand(1, 32, 48, 3)
        request_id = store.save_latent(self.samples)
        self.peer.fail_cleanup = True
        with self.assertRaises(RuntimeError):
            self.run_transfer(request_id=request_id)
        self.assertTrue(store.media_path(request_id, self.options).exists())

    def test_public_shared_output_is_tensor_and_legacy_file_socket_is_blocked(self):
        import sys
        class Blocker:
            def __init__(self, message):
                self.message = message
        self.native.return_value = torch.rand(1, 32, 48, 3)
        with patch.object(service, "peer_api", return_value=self.peer), \
             patch.dict(sys.modules, {"comfy_execution.graph_utils": SimpleNamespace(ExecutionBlocker=Blocker)}):
            output = service.RemoteVAEDecode().run(self.samples, "test.vae", output_mode="shared", memory_policy="keep")
        self.assertTrue(torch.equal(output["result"][0], self.native.return_value))
        self.assertIsInstance(output["result"][1], Blocker)
        self.assertEqual(output["ui"], {"shared_transfer": ["image"]})

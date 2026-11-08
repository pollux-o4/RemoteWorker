"""Real caller/worker/storage lifecycle; replace only HTTP and native encoding."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from remoteworker.vae import (
    codec,
    delivery,
    schema,
    service,
    worker,
)
from remoteworker.vae import (
    shared_store as store,
)
from remoteworker.vae import (
    shared_video as video,
)


class Peer:
    fail_history = False

    def require_node(self, name):
        return {"input": {"required": {"vae_name": [["video.vae", "audio.vae"]]}}}

    def _json(self, path, *, payload):
        return delivery.receive_probe(**payload)

    def submit(self, graph):
        self.graph = graph
        return "peer-job"

    def wait_for_history(self, prompt_id):
        result = worker.SharedVAEVideoWorker().run(**self.graph["save"]["inputs"])
        if self.fail_history:
            raise TimeoutError("connection lost after video saved")
        return {
            prompt_id: {
                "status": {"status_str": "success"},
                "outputs": {"save": result["ui"]},
            }
        }

    def cancel_prompt(self, prompt_id):
        self.cancelled = prompt_id


def encode(samples, path, options):
    path.write_bytes(b"encoded-video")
    return {"frames": 5}


class SharedVideoTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, REMOTEWORKER_SHARED_VAE_DIR=directory))
        self.encoder = self.enterContext(patch.object(worker, "encode_video", side_effect=encode))
        self.peer = Peer()
        self.enterContext(patch.object(video, "peer_api", return_value=self.peer))
        self.samples = {
            "samples": torch.ones(1, 4, 2, 2),
            "noise_mask": torch.ones(1, 2, 2),
        }
        self.request_id = video.SaveSharedVAELatent().run(self.samples)["result"][0]
        self.options = dict(vae_name="video.vae", fps=24.0, audio_vae_name="", crf=18, tiled=False)

    @property
    def latent_file(self):
        return store.request_path("temp", self.request_id, "latent.pt")

    def run_caller(self):
        result = video.RemoteVAEVideo().run(
            **self.options,
            latent=self.request_id,
            server_url="http://peer:8188",
            memory_policy="keep",
        )
        return Path(result["result"][0])

    def test_success_deletes_only_latent_and_completed_retry_skips_encoding(self):
        unrelated = self.latent_file.with_name("keep.txt")
        unrelated.write_text("keep")
        path = self.run_caller()
        self.assertTrue(path.is_file() and unrelated.exists())
        self.assertFalse(self.latent_file.exists())
        self.assertEqual(self.run_caller(), path)
        self.assertEqual(self.encoder.call_count, 1)

    def test_encode_failure_removes_partial_output_and_preserves_input(self):
        def fail(samples, path, options):
            path.write_bytes(b"partial-video")
            raise OSError("encoder failed")

        self.encoder.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, self.request_id):
            self.run_caller()
        self.assertTrue(self.latent_file.is_file())
        output = store.request_path("outputs", self.request_id, "video.mp4")
        self.assertEqual(list(output.parent.iterdir()), [])

    def test_lost_ack_retries_existing_commit_then_deletes_input(self):
        self.peer.fail_history = True
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            self.run_caller()
        self.assertEqual(self.peer.cancelled, "peer-job")
        self.assertTrue(self.latent_file.is_file())
        self.peer.fail_history = False
        self.run_caller()
        self.assertFalse(self.latent_file.exists())
        self.assertEqual(self.encoder.call_count, 1)

    def test_invalid_commits_never_delete_input(self):
        self.peer.fail_history = True
        with self.assertRaises(RuntimeError):
            self.run_caller()
        receipt = store.read_receipt(self.request_id)
        self.peer.fail_history = False
        self.options["fps"] = 30.0
        with self.assertRaisesRegex(RuntimeError, "설정"):
            self.run_caller()
        self.assertEqual(self.encoder.call_count, 1)
        self.options["fps"] = 24.0
        for reported, options in (
            ({**receipt, "path": "/other.mp4"}, self.options),
            (receipt, {**self.options, "fps": 30.0}),
        ):
            with (
                self.subTest(reported=reported, options=options),
                self.assertRaises(ValueError),
            ):
                store.acknowledge(self.request_id, reported, options)
            self.assertTrue(self.latent_file.is_file())
        Path(receipt["path"]).unlink()
        with self.assertRaises(ValueError):
            store.acknowledge(self.request_id, receipt, self.options)
        self.assertTrue(self.latent_file.is_file())

    def test_peer_preflight_failure_reports_retry_id_without_encoding(self):
        self.options["vae_name"] = "missing.vae"
        with self.assertRaisesRegex(RuntimeError, self.request_id):
            self.run_caller()
        self.assertTrue(self.latent_file.is_file())
        self.encoder.assert_not_called()

    def test_shared_storage_preserves_metadata_audio_and_bypasses_http_budget(self):
        audio = {"waveform": torch.ones(1, 2, 100), "sample_rate": 32000}
        with patch.object(codec, "MAX_BYTES", 1):
            self.request_id = store.save_latent(self.samples, audio)
            ticket, loaded = video.LoadSharedVAELatent().run(self.request_id)
            self.assertEqual(ticket, self.request_id)
            for name, value in self.samples.items():
                self.assertTrue(torch.equal(loaded[name], value))
            with self.assertRaisesRegex(ValueError, "transfer limit"):
                codec.dumps(self.samples)
        self.run_caller()
        actual = self.encoder.call_args.args[0]["remote_vae_audio"]
        self.assertTrue(torch.equal(actual["waveform"], audio["waveform"]))

    def test_storage_rejects_missing_configuration_traversal_and_symlinks(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "REMOTEWORKER_SHARED_VAE_DIR"),
        ):
            store.shared_root()
        for value in ("../escape", "", "a" * 33):
            with self.subTest(value=value), self.assertRaises(ValueError):
                store.load_latent(value)
        self.latent_file.unlink()
        self.latent_file.symlink_to("/unrelated/latent.pt")
        with self.assertRaisesRegex(ValueError, "symlink"):
            store.load_latent(self.request_id)

    def test_video_validation_uses_schema_limits(self):
        for name in ("fps", "crf"):
            limits = schema.video_fields()[name][1]
            for value in (limits["min"], limits["max"]):
                self.assertEqual(schema.video_options(**(self.options | {name: value}))[name], value)
            for value in (limits["min"] - 1, limits["max"] + 1, float("nan"), True):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    schema.video_options(**(self.options | {name: value}))


class DeliveryTests(unittest.TestCase):
    setUp = SharedVideoTests.setUp

    def test_storage_handshake_and_early_validation(self):
        self.enterContext(patch.object(service, "peer_api", return_value=self.peer))
        self.assertTrue(service.RemoteVAEDecode.VALIDATE_INPUTS("video.vae", "shared", "http://peer"))
        self.assertTrue(delivery.check_shared(self.peer)["ready"])
        self.assertEqual(list(store.shared_root().glob("temp/*/probe*.json")), [])
        with patch.object(self.peer, "_json", side_effect=RuntimeError("wrong mount")):
            error = service.RemoteVAEDecode.VALIDATE_INPUTS("video.vae", "shared", "http://peer")
        self.assertIsInstance(error, str)
        self.assertIn("공유 폴더 연결 실패", error)
        self.encoder.assert_not_called()

    def test_retry_does_not_request_sampler_and_tensor_mode_preserves_legacy(self):
        node = service.RemoteVAEDecode()
        self.assertEqual(node.check_lazy_status(request_id=self.request_id, output_mode="shared"), [])
        self.assertEqual(
            node.check_lazy_status(request_id=self.request_id, output_mode="tensor"),
            ["samples"],
        )
        self.assertEqual(node.RETURN_TYPES, ("IMAGE", "STRING"))
        self.assertEqual(service.RemoteVAEDecodeAudio.RETURN_TYPES, ("AUDIO", "STRING"))
        self.assertEqual(
            schema.decode_inputs("image")["optional"]["output_mode"][1]["default"],
            "shared",
        )

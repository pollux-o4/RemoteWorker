"""Adversarial checks around native publication and destructive acknowledgments."""

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from remoteworker.vae import codec, shared_store as store, schema, worker


class NativeVaeLifecycle(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(
            patch.dict(os.environ, REMOTEWORKER_SHARED_VAE_DIR=str(self.root))
        )
        self.samples = {"samples": torch.ones(1, 4, 2, 2)}
        self.options = schema.video_options("video.vae", "", 24.0, 18, False)

    def test_concurrent_publish_commits_once_and_reuses_same_receipt(self):
        request_id = store.save_latent(self.samples)
        started, release = threading.Event(), threading.Event()
        calls, receipts, errors = [], [], []

        def encode(samples, path, options):
            calls.append(True)
            started.set()
            if not release.wait(3):
                raise TimeoutError("encoder not released")
            path.write_bytes(b"encoded-video")
            return {"frames": 5}

        def publish():
            try:
                receipts.append(store.publish(request_id, self.options, encode))
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=publish)
        second = threading.Thread(target=publish)
        first.start()
        self.assertTrue(started.wait(3))
        second.start()
        release.set()
        for thread in (first, second):
            thread.join(4)
        self.assertFalse(any(thread.is_alive() for thread in (first, second)))
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(receipts), 2)
        self.assertEqual(receipts[0], receipts[1])
        self.assertTrue(store.request_path("temp", request_id, "latent.pt").is_file())

    def test_path_components_and_dangling_symlink_do_not_escape_owned_root(self):
        request_id = "a" * 32
        for area, name in (
            ("../other", "latent.pt"),
            ("temp", "../escape"),
            ("outputs", "/tmp/out"),
        ):
            with self.subTest(area=area, name=name), self.assertRaises(ValueError):
                store.request_path(area, request_id, name)
        directory = self.root / "temp" / request_id
        directory.mkdir(parents=True)
        (directory / "latent.pt").symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "symlink"):
            store.request_path("temp", request_id, "latent.pt")

    def test_tampered_receipt_never_acknowledges_or_removes_latent(self):
        request_id = store.save_latent(self.samples)

        def encode(samples, path, options):
            path.write_bytes(b"video")
            return {"frames": 5}

        receipt = store.publish(request_id, self.options, encode)
        file = store.receipt_file(request_id)
        value = json.loads(file.read_text())
        value["bytes"] += 1
        file.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            store.acknowledge(request_id, receipt, self.options)
        self.assertTrue(store.request_path("temp", request_id, "latent.pt").is_file())

    def test_audio_sample_rate_bool_is_not_a_valid_integer_rate(self):
        with self.assertRaises(ValueError):
            codec.validate_result(
                {"waveform": torch.ones(1, 2, 4), "sample_rate": True}, "audio"
            )

    def test_output_budget_saturates_instead_of_wrapping_large_shapes(self):
        from types import SimpleNamespace

        vae = SimpleNamespace(upscale_ratio=(2, 2), output_channels=3)
        samples = {"samples": SimpleNamespace(shape=(2**32, 4, 2**32, 2**32))}
        with self.assertRaisesRegex(ValueError, "budget"):
            worker.check_output_budget(vae, samples, "image")

    def test_tensor_metadata_sequence_subtypes_survive_cpu_and_restore(self):
        value = {"shape": torch.Size([2, 3])}
        packed = codec.cpu_value(value)
        self.assertIsInstance(packed["shape"], torch.Size)
        self.assertIsInstance(codec.restore(packed)["shape"], torch.Size)

"""Adversarial checks around native publication and destructive acknowledgments."""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from remoteworker.vae import codec, schema, worker
from remoteworker.vae import shared_store as store


class NativeVaeLifecycle(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, REMOTEWORKER_SHARED_VAE_DIR=str(self.root)))
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
            codec.validate_result({"waveform": torch.ones(1, 2, 4), "sample_rate": True}, "audio")

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

    def test_publication_directory_swap_cannot_redirect_writer_to_outside(self):
        request_id = "b" * 32
        target = store.request_path("outputs", request_id, "video.mp4")
        target.parent.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        victim = outside / target.name
        victim.write_bytes(b"keep")
        pinned = target.parent.with_name(request_id + "-moved")

        def swap(temporary):
            target.parent.rename(pinned)
            target.parent.symlink_to(outside, target_is_directory=True)
            temporary.write_bytes(b"encoded")

        with self.assertRaises((ValueError, OSError)):
            store.atomic_write(target, swap)
        self.assertEqual(victim.read_bytes(), b"keep")
        self.assertEqual(list(outside.iterdir()), [victim])
        self.assertEqual(list(pinned.iterdir()), [])

    def test_temporary_name_swap_cannot_redirect_writer_to_symlink_target(self):
        target = store.request_path("outputs", "c" * 32, "video.mp4")
        target.parent.mkdir(parents=True)
        victim = self.root / "victim.mp4"
        victim.write_bytes(b"keep")

        def swap(temporary):
            name = next(target.parent.glob(".video-*"))
            name.unlink()
            name.symlink_to(victim)
            temporary.write_bytes(b"encoded")

        with self.assertRaises((ValueError, OSError)):
            store.atomic_write(target, swap)
        self.assertEqual(victim.read_bytes(), b"keep")
        self.assertEqual(list(target.parent.iterdir()), [])

    def test_tiled_options_reject_coercion_types_and_out_of_range_before_io(self):
        from types import SimpleNamespace

        from remoteworker.vae import service, shared_tensor

        invalid = {
            "tiled": ["false", 0, 1, None, []],
            "tile_size": [0, -1, 4097, True, "512", 512.0],
            "overlap": [-1, 4097, False, "64", 64.0],
            "temporal_size": [0, 7, 4097, True, "64"],
            "temporal_overlap": [0, 3, 4097, True, "8"],
        }
        api = SimpleNamespace(require_node=lambda *_: self.fail("invalid options reached peer"))
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        shared_tensor.options("video.vae", "image", **{field: value})
                    with self.assertRaises(ValueError):
                        service.execute_decode(
                            api,
                            samples=self.samples,
                            vae_name="video.vae",
                            kind="image",
                            options={field: value},
                            memory_policy="keep",
                        )
        for value in invalid["tiled"]:
            with self.assertRaises(ValueError):
                schema.video_options("video.vae", "", 24.0, 18, value)
        # Native VAE reduces these overlaps; validation must not reject supported settings.
        settings = shared_tensor.options(
            "video.vae",
            "image",
            tile_size=64,
            overlap=4096,
            temporal_size=8,
            temporal_overlap=4096,
        )
        self.assertEqual(settings["overlap"], 4096)
        self.assertEqual(settings["temporal_overlap"], 4096)

    def test_shared_cancellation_and_chained_retry_error_survive_cleanup_failure(self):
        from remoteworker.core.cancellation import RequestCancelled
        from remoteworker.vae import delivery, shared_tensor, shared_video

        class InterruptProcessingException(Exception):
            pass

        owner = self

        class Peer:
            def require_node(self, name):
                return {"input": {"required": {"vae_name": [["video.vae"]]}}}

            def _json(self, path, *, payload):
                return delivery.receive_probe(**payload)

            def submit(self, graph):
                return "remote-job"

            def wait_for_history(self, prompt_id):
                raise self.failure

            def cancel_prompt(self, prompt_id):
                owner.assertEqual(prompt_id, "remote-job")
                self.cancelled = True
                raise OSError("cleanup unavailable")

        for shared_tensor_mode in (False, True):
            for kind in (RequestCancelled, InterruptProcessingException, TimeoutError):
                with self.subTest(tensor=shared_tensor_mode, exception=kind):
                    peer = Peer()
                    failure = peer.failure = kind("original failure")
                    request_id = store.save_latent(self.samples)
                    expected = RuntimeError if kind is TimeoutError else kind
                    with self.assertLogs("RemoteWorker", level="WARNING"):
                        try:
                            if shared_tensor_mode:
                                shared_tensor.execute(
                                    peer,
                                    self.samples,
                                    "video.vae",
                                    "image",
                                    "http://peer",
                                    "keep",
                                    {},
                                    request_id,
                                )
                            else:
                                shared_video.execute_shared(
                                    peer,
                                    request_id,
                                    self.options,
                                    server_url="http://peer",
                                    memory_policy="keep",
                                )
                            self.fail("expected shared execution failure")
                        except expected as error:
                            caught = error
                            original_traceback = error.__traceback__
                    self.assertTrue(peer.cancelled)
                    if kind is TimeoutError:
                        self.assertIs(caught.__cause__, failure)
                    else:
                        self.assertIs(caught, failure)
                        self.assertIn(request_id, " ".join(failure.__notes__))
                    self.assertIsNotNone(original_traceback)
                    self.assertTrue(store.request_path("temp", request_id, "latent.pt").is_file())

    def test_cancel_after_tensor_validation_preserves_all_consumption_files(self):
        from remoteworker.core.cancellation import RequestCancelled
        from remoteworker.vae import shared_tensor

        request_id = store.save_latent(self.samples)
        options = shared_tensor.options("video.vae", "image")

        def encode(samples, path, options):
            torch.save(torch.ones(1, 4, 4, 3), path)
            return {"kind": "image"}

        receipt = store.publish(request_id, options, encode)
        failure = RequestCancelled("cancel before acknowledgment")
        with patch(
            "remoteworker.vae.shared_tensor.check_cancelled",
            side_effect=[None, failure],
        ):
            with self.assertRaises(RequestCancelled) as caught:
                shared_tensor.consume(request_id, receipt, options)
        self.assertIs(caught.exception, failure)
        for path in (
            store.request_path("temp", request_id, "latent.pt"),
            store.media_path(request_id, options),
            store.receipt_file(request_id, options),
        ):
            self.assertTrue(path.is_file())

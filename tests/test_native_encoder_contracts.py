"""Rust boundary contracts for encoder plans and profile policy."""

import unittest

from remoteworker import _native
from remoteworker.core.protocol import ClipSelection
from remoteworker.qwen.service import build_workflow
from remoteworker.prompt.profiles import validate_profile_payload, DEFAULT_PROFILE


class NativeEncoderContracts(unittest.TestCase):
    def test_qwen_resolution_rejects_bool_before_building_graph(self):
        for resolution in (True, -1, 4097, 1.5):
            with (
                self.subTest(resolution=resolution),
                self.assertRaisesRegex(ValueError, "resolution"),
            ):
                build_workflow(
                    clip=ClipSelection("clip", "qwen_image", "default"),
                    vae_name="vae",
                    prompt="edit",
                    negative_prompt="",
                    resolution=resolution,
                    negative_vision=False,
                    image_refs=[],
                    request_id="request",
                )

    def test_explicit_null_memory_policy_cannot_become_release_default(self):
        with self.assertRaisesRegex(ValueError, "peer_memory_policy"):
            build_workflow(
                clip=ClipSelection("clip", "qwen_image", "default"),
                vae_name="vae",
                prompt="edit",
                negative_prompt="",
                resolution=1024,
                negative_vision=False,
                image_refs=[],
                request_id="request",
                peer_memory_policy=None,
            )

    def test_profile_media_requires_array_at_python_boundary(self):
        payload = DEFAULT_PROFILE.as_public_dict()
        payload["accepted_media"] = ("image",)
        with self.assertRaisesRegex(ValueError, "array"):
            validate_profile_payload(payload)

    def test_qwen_dimensions_use_native_round_to_even_and_positive_grid(self):
        self.assertEqual((64, 64), _native.qwen_resize_native(80, 80, 0))
        self.assertEqual((128, 64), _native.qwen_resize_native(128, 64, 0))
        with self.assertRaises(ValueError):
            _native.qwen_resize_native(0, 64, 1024)

    def test_profile_rejects_boolean_budget_and_video_on_non_minimax(self):
        payload = DEFAULT_PROFILE.as_public_dict()
        payload["max_length"] = True
        with self.assertRaisesRegex(ValueError, "max_length"):
            validate_profile_payload(payload)
        payload["max_length"] = 256
        payload["accepted_media"] = ["video"]
        with self.assertRaisesRegex(ValueError, "minimax"):
            validate_profile_payload(payload)

    def test_profile_registry_rejects_duplicate_ids(self):
        profile = DEFAULT_PROFILE.as_public_dict()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _native.profile_registry_native(
                {"version": 1, "profiles": [profile, profile]},
                DEFAULT_PROFILE.system_prompt,
                list(DEFAULT_PROFILE.accepted_media),
            )

    def test_sparse_and_alias_sockets_rejected(self):
        for media in (
            {"ref_image_1": object()},
            {"ref_image_0": object(), "ref_image_00": object()},
        ):
            with self.assertRaisesRegex(ValueError, "gaps"):
                _native.connected_slots_native("ref_image", media, 9)

    def test_cache_identity_excludes_peer_cleanup_policy_and_normalizes_origin(self):
        values = dict(
            clip=ClipSelection("clip", "minimax", "default"),
            vae_name="vae",
            prompt="test",
            width=64,
            height=64,
            length=5,
            first_frame=None,
            last_frame=None,
            server_url="http://peer///",
        )
        key = _native.cache_identity_native("minimax-i2v", values)
        self.assertEqual("http://peer", key[-1])
        self.assertEqual("minimax-h3-i2v", key[0])

    def test_prompt_artifact_is_bounded_utf8_and_null_free(self):
        for data in (b"", b"\xff", b"prompt\0", b"x" * (128 * 1024 + 1)):
            with self.assertRaises(RuntimeError):
                _native.decode_generated_prompt_native(data)
        self.assertEqual("hello", _native.decode_generated_prompt_native(b" hello\n"))

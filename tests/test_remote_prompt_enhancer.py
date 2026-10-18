from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT))
import remoteworker.prompt.profiles as profiles
import remoteworker.prompt.service as prompt_enhancer


class PromptEnhancerProfileStoreTests(unittest.TestCase):
    def test_profiles_are_nvi_local_and_support_described_upsert_list_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = profiles.PromptEnhancerProfileStore(Path(directory) / "profiles.json")
            self.assertEqual(["example"], [profile.id for profile in store.list()])
            stored = store.upsert({
                "id": "portrait-detail",
                "description": "Preserve a portrait identity while adding composition detail.",
                "clip_name": "qwen3vl_4b_fp8_scaled(krea).safetensors",
                "clip_type": "krea2",
                "device": "default",
                "max_length": 320,
                "system_prompt": "Return a concise English prompt grounded in the references.",
                "accepted_media": ["image"],
            })
            self.assertEqual("portrait-detail", stored.id)
            self.assertEqual(("image",), stored.accepted_media)
            self.assertEqual(
                ["example", "portrait-detail"],
                [profile.id for profile in store.list()],
            )
            self.assertTrue(store.delete("portrait-detail"))
            self.assertFalse(store.delete("portrait-detail"))
            self.assertEqual(["example"], [profile.id for profile in store.list()])

    def test_invalid_profile_fails_before_any_registry_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            store = profiles.PromptEnhancerProfileStore(path)
            with self.assertRaisesRegex(ValueError, "profile id"):
                store.upsert({
                    "id": "Not Safe", "description": "x", "clip_name": "a", "clip_type": "krea2",
                    "device": "default", "max_length": 256,
                    "system_prompt": "x", "accepted_media": ["image"],
                })
            self.assertFalse(path.exists())

    def test_profile_accepts_a_long_user_authored_system_prompt(self) -> None:
        profile = profiles.validate_profile_payload({
            "id": "long-instructions", "description": "Long prompt profile.",
            "clip_name": "qwen.safetensors", "clip_type": "krea2", "device": "default",
            "max_length": 1024, "system_prompt": "instruction " * 10_000,
            "accepted_media": ["image"],
        })
        self.assertGreater(len(profile.system_prompt), 8192)

    def test_legacy_profile_is_upgraded_and_video_requires_a_minimax_profile(self) -> None:
        legacy = profiles.validate_profile_payload({
            "id": "legacy", "description": "Old profile.", "clip_name": "qwen.safetensors",
            "clip_type": "krea2", "device": "default", "max_length": 256,
        })
        self.assertEqual(("image",), legacy.accepted_media)
        with self.assertRaisesRegex(ValueError, "video support requires clip_type minimax"):
            profiles.validate_profile_payload({
                "id": "wrong-video", "description": "x", "clip_name": "qwen.safetensors",
                "clip_type": "krea2", "device": "default", "max_length": 256,
                "system_prompt": "x", "accepted_media": ["image", "video"],
            })


class RemotePromptEnhancerExecutionTests(unittest.TestCase):
    def test_executes_a_server_resolved_profile_and_returns_only_the_text_artifact(self) -> None:
        class FakeApi:
            def __init__(self):
                self.nodes = []
                self.workflow = None

            def get_prompt_enhancer_profile(self, profile_id):
                if profile_id != "example":
                    raise AssertionError(profile_id)
                return profiles.DEFAULT_PROFILE

            def require_node(self, node_name):
                self.nodes.append(node_name)
                return {}

            def submit(self, workflow):
                self.workflow = workflow
                return "prompt-1"

            def wait_for_history(self, _prompt_id):
                return {"prompt-1": {"status": {"status_str": "success"}, "outputs": {
                    "save": {"text": ["remote-conditioning/request-1/prompt_00001_.txt"]},
                }}}

            def download_artifact(self, artifact):
                if artifact != "remote-conditioning/request-1/prompt_00001_.txt":
                    raise AssertionError(artifact)
                return b"cinematic portrait, soft daylight"

        api = FakeApi()
        output = prompt_enhancer.execute(
            profile_id="example", prompt="a student in a library", request_id="request-1", api=api,
        )
        self.assertEqual("cinematic portrait, soft daylight", output)
        self.assertEqual(["RemotePromptGenerate", "SaveRemoteText"], api.nodes)
        self.assertEqual("krea2", api.workflow["clip"]["inputs"]["type"])
        self.assertEqual("qwen-image", api.workflow["generate"]["inputs"]["media_strategy"])
        self.assertEqual(profiles.DEFAULT_PROFILE.system_prompt, api.workflow["generate"]["inputs"]["system_prompt"])
        self.assertIn("a student in a library", api.workflow["generate"]["inputs"]["prompt"])

    def test_rejects_empty_prompt_before_contacting_the_peer(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            prompt_enhancer.build_generation_prompt("   ")

    def test_node_schema_keeps_normal_use_compact(self) -> None:
        inputs = prompt_enhancer.RemotePromptEnhancer.INPUT_TYPES()
        self.assertEqual({"prompt", "profile_id", "server_url", "timeout_seconds"}, set(inputs["required"]))
        self.assertEqual(("IMAGE",), inputs["optional"]["image_0"])
        dynamic = prompt_enhancer.RemotePromptEnhancer.INPUT_TYPES()["optional"]
        self.assertIn("image_0", dynamic)
        self.assertIn("video_0", dynamic)

    def test_workflow_binds_reference_batches_only_when_profile_allows_them(self) -> None:
        minimax = profiles.PromptEnhancerProfile(
            id="minimax", description="Video aware.", clip_name="minimax.safetensors", clip_type="minimax",
            device="default", max_length=256, system_prompt="Ground the result in each reference.",
            accepted_media=("image", "video"),
        )
        workflow = prompt_enhancer.build_workflow(
            minimax, "A short action scene", "request-1", image_refs=["one.png [temp]", "two.png [temp]"],
            video_refs=[["frame-1.png [temp]", "frame-2.png [temp]"], ["frame-3.png [temp]", "frame-4.png [temp]"]],
        )
        self.assertEqual("minimax-ref2va", workflow["generate"]["inputs"]["media_strategy"])
        self.assertEqual(["image_batch_1", 0], workflow["generate"]["inputs"]["image"])
        self.assertEqual(["video_0_batch_1", 0], workflow["generate"]["inputs"]["video_0"])
        self.assertEqual(["video_1_batch_1", 0], workflow["generate"]["inputs"]["video_1"])
        self.assertEqual("ImageBatch", workflow["video_1_batch_1"]["class_type"])

    def test_nvi_generation_uses_the_profile_system_prompt_and_correct_tokenizer_media_contract(self) -> None:
        class Batch:
            def __init__(self, size):
                self.shape = (size, 16, 16, 3)

            def __getitem__(self, value):
                return ("slice", value.start, value.stop)

        class Clip:
            def __init__(self):
                self.calls = []

            def tokenize(self, text, **kwargs):
                self.calls.append((text, kwargs))
                return "tokens"

            def generate(self, tokens, **kwargs):
                self.calls.append((tokens, kwargs))
                return "generated"

            def decode(self, generated):
                return "enhanced"

        clip = Clip()
        result = prompt_enhancer.RemotePromptGenerate().run(
            clip, "keep the red coat", "Write only the final English prompt.", "qwen-image", 256, image=Batch(1),
        )
        self.assertEqual(("enhanced",), result)
        self.assertIn("Write only the final English prompt.", clip.calls[0][0])
        self.assertEqual(Batch, type(clip.calls[0][1]["image"]))

        clip = Clip()
        prompt_enhancer.RemotePromptGenerate().run(
            clip, "show the transition", "Use every reference.", "minimax-ref2va", 256,
            image=Batch(2), video_0=Batch(6), video_1=Batch(4),
        )
        refs = clip.calls[0][1]["minimax_ref_items"]
        self.assertEqual(["image", "image", "video", "video"], [item["type"] for item in refs])
        self.assertEqual(("slice", 0, 1), refs[0]["data"])
        self.assertEqual(6, refs[-2]["data"].shape[0])
        self.assertEqual(4, refs[-1]["data"].shape[0])

    def test_timeout_and_caller_interrupt_cancel_the_submitted_prompt_enhancer_job(self) -> None:
        class CallerInterrupted(BaseException):
            pass

        class Api:
            def __init__(self, failure):
                self.failure = failure
                self.cancelled = []

            def get_prompt_enhancer_profile(self, _profile_id):
                return profiles.DEFAULT_PROFILE

            def require_node(self, _node_name):
                return {}

            def submit(self, _workflow):
                return "prompt-1"

            def wait_for_history(self, _prompt_id):
                raise self.failure

            def cancel_prompt(self, prompt_id):
                self.cancelled.append(prompt_id)

        for failure in (TimeoutError("peer wait expired"), CallerInterrupted()):
            api = Api(failure)
            with self.assertRaises(type(failure)):
                prompt_enhancer.execute(
                    profile_id="example", prompt="a student in a library", request_id="request-1", api=api,
                )
            self.assertEqual(["prompt-1"], api.cancelled)


if __name__ == "__main__":
    unittest.main()


class PromptMediaUploadTests(unittest.TestCase):
    def test_executes_with_image_and_video_and_discards_its_request(self):
        from dataclasses import replace
        from unittest.mock import Mock, patch

        import torch
        api = Mock()
        api.get_prompt_enhancer_profile.return_value = replace(profiles.DEFAULT_PROFILE, accepted_media=("image", "video"))
        api.submit.return_value = "prompt-1"
        api.upload_image.return_value = "remote/request-1/image.png [temp]"
        api.upload_file.return_value = "remote/request-1/video.png [temp]"
        api.wait_for_history.return_value = {"prompt-1": {
            "status": {"status_str": "success"},
            "outputs": {"save": {"text": ["remote-conditioning/request-1/prompt.txt"]}},
        }}
        api.download_artifact.return_value = b"generated prompt"
        with patch.object(prompt_enhancer, "encode_image_frame_png", return_value=b"png"):
            result = prompt_enhancer.execute(profile_id="profile", prompt="scene", request_id="request-1", api=api,
                images=[torch.zeros(2, 2, 3)], videos=[torch.zeros(2, 2, 2, 3)])
        self.assertEqual("generated prompt", result)
        self.assertEqual(1, api.upload_image.call_count)
        self.assertEqual(2, api.upload_file.call_count)
        api.discard_inputs.assert_called_once_with("request-1")
        api.cancel_prompt.assert_not_called()

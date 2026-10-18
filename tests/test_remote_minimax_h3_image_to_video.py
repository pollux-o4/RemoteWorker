from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import remoteworker.minimax.image_to_video as remote_i2v


class FakeImage:
    shape = (1, 64, 96, 3)

    def __getitem__(self, index):
        return (self, index)


class MiniMaxImageToVideoContractTests(unittest.TestCase):
    def test_schema_mirrors_native_fixed_frame_contract(self) -> None:
        schema = remote_i2v.RemoteMiniMaxH3ImageToVideo.INPUT_TYPES()
        self.assertIn("prompt", schema["required"])
        self.assertEqual("", schema["required"]["clip_name"][0][0])
        self.assertEqual(("IMAGE",), schema["optional"]["first_frame"])
        self.assertEqual(("IMAGE",), schema["optional"]["last_frame"])
        self.assertEqual(("*", {"forceInput": True}), schema["optional"]["vae_name_override"])
        self.assertEqual({"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}, schema["hidden"])
        self.assertNotIn("peer_memory_policy", schema["required"])
        self.assertEqual(["release_after_result", "keep"], schema["optional"]["peer_memory_policy"][0])
        self.assertEqual(["reuse_same_inputs", "always_refresh"], schema["optional"]["cache_policy"][0])
        self.assertNotIn("ref_image_count", schema["required"])

    def test_cache_policy_uses_native_signature_by_default_and_nan_only_for_refresh(self) -> None:
        self.assertEqual(0, remote_i2v.RemoteMiniMaxH3ImageToVideo.IS_CHANGED(cache_policy="reuse_same_inputs"))
        self.assertTrue(math.isnan(remote_i2v.RemoteMiniMaxH3ImageToVideo.IS_CHANGED(cache_policy="always_refresh")))

    def test_validate_inputs_accepts_remote_discovered_model_aliases(self) -> None:
        result = remote_i2v.RemoteMiniMaxH3ImageToVideo.VALIDATE_INPUTS(
            clip_name="hot-tier\\future-minimax-clip.safetensors",
            clip_type="minimax",
            device="default",
            vae_name="hot-tier\\future-video.vae",
        )

        self.assertTrue(result)

    def test_validate_inputs_rejects_empty_loader_values(self) -> None:
        result = remote_i2v.RemoteMiniMaxH3ImageToVideo.VALIDATE_INPUTS(
            clip_name="minimax.safetensors",
            clip_type="",
            device="default",
            vae_name="video.vae",
        )

        self.assertIn("clip_type", result)

    def test_validate_inputs_rejects_invalid_cache_policy(self) -> None:
        result = remote_i2v.RemoteMiniMaxH3ImageToVideo.VALIDATE_INPUTS(
            clip_name="minimax.safetensors",
            clip_type="minimax",
            device="default",
            vae_name="video.vae",
            cache_policy="disk_cache",
        )

        self.assertIn("cache_policy", result)

    def test_workflow_maps_optional_frames_to_exact_native_input_names(self) -> None:
        workflow = remote_i2v.build_workflow(
            clip=remote_i2v.ClipSelection("minimax.safetensors", "minimax", "default"),
            vae_name="video.vae", prompt="a moving scene", width=1344, height=768, length=124,
            first_frame="remote/request/first.png [temp]", last_frame="remote/request/last.png [temp]", request_id="request-1",
        )
        inputs = workflow["encode"]["inputs"]
        self.assertEqual("MiniMaxH3ImageToVideo", workflow["encode"]["class_type"])
        self.assertEqual(["first_frame_loader", 0], inputs["first_frame"])
        self.assertEqual(["last_frame_loader", 0], inputs["last_frame"])
        self.assertEqual("SaveRemoteLatent", workflow["save_latent"]["class_type"])
        self.assertEqual(["save", 0], workflow["save_latent"]["inputs"]["dependency"])
        self.assertEqual("VRAMCleanup", workflow["cleanup"]["class_type"])
        self.assertEqual(["save_latent", 0], workflow["cleanup"]["inputs"]["anything"])
        self.assertFalse(workflow["cleanup"]["inputs"]["reset_execution_cache"])

    def test_keep_policy_does_not_add_cleanup_or_dependency_edge(self) -> None:
        workflow = remote_i2v.build_workflow(
            clip=remote_i2v.ClipSelection("minimax.safetensors", "minimax", "default"),
            vae_name="video.vae", prompt="a moving scene", width=1344, height=768, length=124,
            first_frame=None, last_frame=None, request_id="request-1", peer_memory_policy="keep",
        )

        self.assertNotIn("cleanup", workflow)
        self.assertNotIn("dependency", workflow["save_latent"]["inputs"])

    def test_execute_uploads_connected_frames_only_and_discards_them(self) -> None:
        class FakeApi:
            def __init__(self):
                self.events = []
                self.workflow = None

            def require_node(self, name):
                self.events.append(("capability", name))
                if name == "VRAMCleanup":
                    return {
                        "input": {
                            "required": {"offload_model": [], "offload_cache": [], "reset_execution_cache": []},
                            "optional": {"anything": ["*"]},
                        },
                        "output": ["*"],
                    }
                if name == "VAELoader":
                    return {"input": {"required": {"vae_name": [["video.vae"]]}}}
                return {"output": ["CONDITIONING", "LATENT"]}

            def clip_loader_options(self):
                return {"clip_name": ["minimax.safetensors"], "clip_type": ["minimax"], "device": ["default"]}

            def upload_file(self, **kwargs):
                self.events.append(("file", kwargs["filename"], kwargs["content_type"]))
                return f"remote-conditioning/request-1/{kwargs['filename']} [temp]"

            def submit(self, workflow):
                self.workflow = workflow
                return "prompt-1"

            def wait_for_history(self, _prompt_id):
                return {"prompt-1": {"status": {"status_str": "success"}, "outputs": {
                    "save": {"text": ["remote-conditioning/request-1/cond_00001_.cond"]},
                    "save_latent": {"latents": ["remote-conditioning/request-1/latent_00001_.latent"]},
                }}}

            def download_artifact(self, path):
                return path.encode()

            def discard_inputs(self, request_id):
                self.events.append(("discard", request_id))
                return True

        api = FakeApi()
        with patch.object(remote_i2v, "encode_image_frame_png", return_value=b"png"):
            result = remote_i2v.execute(
                clip=remote_i2v.ClipSelection("minimax.safetensors", "minimax", "default"),
                vae_name="video.vae", prompt="a moving scene", width=1344, height=768, length=124,
                first_frame=FakeImage(), last_frame=None, request_id="request-1", api=api,
                deserialize=lambda cond, latent: (cond, latent),
            )

        self.assertEqual((b"remote-conditioning/request-1/cond_00001_.cond", b"remote-conditioning/request-1/latent_00001_.latent"), result)
        self.assertEqual(("capability", "MiniMaxH3ImageToVideo"), api.events[0])
        self.assertEqual(("capability", "VRAMCleanup"), api.events[1])
        self.assertEqual([("file", "first_frame.png", "image/png")], [event for event in api.events if event[0] == "file"])
        self.assertNotIn("last_frame", api.workflow["encode"]["inputs"])
        self.assertEqual(("discard", "request-1"), api.events[-1])

    def test_connected_vae_name_uses_primary_widget_without_override_socket(self) -> None:
        node = remote_i2v.RemoteMiniMaxH3ImageToVideo()
        with patch.object(remote_i2v, "execute", return_value=("cond", "latent")) as execute:
            result = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name="remote-models\\video.vae",
                width=1344,
                height=768,
                length=124,
            )

        self.assertEqual(("cond", "latent"), result)
        self.assertEqual("remote-models\\video.vae", execute.call_args.kwargs["vae_name"])

    def test_reuses_same_inputs_without_resubmitting_to_peer(self) -> None:
        node = remote_i2v.RemoteMiniMaxH3ImageToVideo()
        calls = []

        def fake_execute(**kwargs):
            calls.append(kwargs["peer_memory_policy"])
            return kwargs["deserialize"](b"cond-bytes", b"latent-bytes")

        with (
            patch.object(remote_i2v, "execute", side_effect=fake_execute),
            patch.object(remote_i2v, "decode_remote_outputs", side_effect=lambda cond, latent: (cond, {"samples": latent})),
        ):
            first = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name="video.vae",
                width=1344,
                height=768,
                length=124,
                peer_memory_policy="release_after_result",
            )
            second = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name="video.vae",
                width=1344,
                height=768,
                length=124,
                peer_memory_policy="keep",
            )

        self.assertEqual((b"cond-bytes", {"samples": b"latent-bytes"}), first)
        self.assertEqual(first, second)
        self.assertEqual(["release_after_result"], calls)

    def test_always_refresh_resubmits_even_with_same_inputs(self) -> None:
        node = remote_i2v.RemoteMiniMaxH3ImageToVideo()
        calls = []

        def fake_execute(**kwargs):
            calls.append(kwargs["request_id"])
            return kwargs["deserialize"](b"cond-bytes", b"latent-bytes")

        with (
            patch.object(remote_i2v, "execute", side_effect=fake_execute),
            patch.object(remote_i2v, "decode_remote_outputs", side_effect=lambda cond, latent: (cond, {"samples": latent})),
        ):
            for _index in range(2):
                node.run(
                    prompt="scene",
                    clip_name="minimax.safetensors",
                    clip_type="minimax",
                    device="default",
                    vae_name="video.vae",
                    width=1344,
                    height=768,
                    length=124,
                    cache_policy="always_refresh",
                )

        self.assertEqual(2, len(calls))

    def test_connected_vae_loader_object_resolves_name_from_workflow_metadata(self) -> None:
        node = remote_i2v.RemoteMiniMaxH3ImageToVideo()
        extra_pnginfo = {
            "workflow": {
                "nodes": [
                    {"id": 119, "type": "VAELoader", "widgets_values": ["remote-models\\video.vae"]},
                    {
                        "id": 150,
                        "type": "RemoteMiniMaxH3ImageToVideo",
                        "inputs": [{"name": "vae_name", "link": 9001}],
                    },
                ],
                "links": [[9001, 119, 0, 150, 0, "VAE"]],
            }
        }

        with patch.object(remote_i2v, "execute", return_value=("cond", "latent")) as execute:
            result = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name=object(),
                width=1344,
                height=768,
                length=124,
                unique_id=150,
                extra_pnginfo=extra_pnginfo,
            )

        self.assertEqual(("cond", "latent"), result)
        self.assertEqual("remote-models\\video.vae", execute.call_args.kwargs["vae_name"])

    def test_legacy_vae_override_socket_remains_backward_compatible(self) -> None:
        node = remote_i2v.RemoteMiniMaxH3ImageToVideo()
        with patch.object(remote_i2v, "execute", return_value=("cond", "latent")) as execute:
            result = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name="dropdown-video.vae",
                width=1344,
                height=768,
                length=124,
                vae_name_override="remote-models\\video.vae",
            )

        self.assertEqual(("cond", "latent"), result)
        self.assertEqual("remote-models\\video.vae", execute.call_args.kwargs["vae_name"])

    def test_missing_cleanup_capability_fails_before_upload(self) -> None:
        class FakeApi:
            def __init__(self):
                self.events = []

            def require_node(self, name):
                self.events.append(("capability", name))
                if name == "VRAMCleanup":
                    return {"input": {"required": {}, "optional": {}}, "output": ["*"]}
                return {"output": ["CONDITIONING", "LATENT"]}

            def upload_file(self, **_kwargs):
                self.events.append("upload")
                raise AssertionError("must fail before upload")

        api = FakeApi()
        with self.assertRaisesRegex(RuntimeError, "VRAMCleanup"):
            remote_i2v.execute(
                clip=remote_i2v.ClipSelection("minimax.safetensors", "minimax", "default"),
                vae_name="video.vae", prompt="a moving scene", width=1344, height=768, length=124,
                first_frame=FakeImage(), last_frame=None, request_id="request-1", api=api,
                deserialize=lambda cond, latent: (cond, latent),
            )

        self.assertEqual([("capability", "MiniMaxH3ImageToVideo"), ("capability", "VRAMCleanup")], api.events)

    def test_invalid_frame_or_dimension_fails_before_upload(self) -> None:
        with self.assertRaisesRegex(ValueError, "BHWC"):
            remote_i2v._single_frame(object(), "first_frame")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            remote_i2v.build_workflow(
                clip=remote_i2v.ClipSelection("minimax.safetensors", "minimax", "default"),
                vae_name="video.vae", prompt="scene", width=33, height=768, length=124,
                first_frame=None, last_frame=None, request_id="request-1",
            )

    def test_timeout_and_caller_interrupt_cancel_the_submitted_i2v_prompt(self) -> None:
        class CallerInterrupted(BaseException):
            pass

        class Api:
            def __init__(self, failure):
                self.failure = failure
                self.cancelled = []

            def require_node(self, name):
                if name == "MiniMaxH3ImageToVideo":
                    return {"output": ["CONDITIONING", "LATENT"]}
                if name == "VAELoader":
                    return {"input": {"required": {"vae_name": [["video.vae"]]}}}
                raise AssertionError(name)

            def clip_loader_options(self):
                return {"clip_name": ["minimax.safetensors"], "clip_type": ["minimax"], "device": ["default"]}

            def submit(self, _workflow):
                return "prompt-1"

            def wait_for_history(self, _prompt_id):
                raise self.failure

            def cancel_prompt(self, prompt_id):
                self.cancelled.append(prompt_id)

        for failure in (TimeoutError("peer wait expired"), CallerInterrupted()):
            api = Api(failure)
            with self.assertRaises(type(failure)):
                remote_i2v.execute(
                    clip=remote_i2v.ClipSelection("minimax.safetensors", "minimax", "default"),
                    vae_name="video.vae", prompt="scene", width=1344, height=768, length=124,
                    first_frame=None, last_frame=None, request_id="request-1", api=api,
                    deserialize=lambda cond, latent: (cond, latent), peer_memory_policy="keep",
                )
            self.assertEqual(["prompt-1"], api.cancelled)


if __name__ == "__main__":
    unittest.main()

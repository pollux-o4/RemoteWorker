from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import remoteworker.minimax.reference as remote_minimax
from remoteworker.minimax import media as reference_media


class FakeImage:
    def __init__(self, frames: int):
        self.shape = (frames, 64, 96, 3)

    def __getitem__(self, index):
        return (self, index)


class MiniMaxReferenceContractTests(unittest.TestCase):
    def test_dynamic_socket_families_match_native_ref2va_contract(self) -> None:
        sockets = reference_media.DynamicReferenceMedia()
        self.assertEqual(("IMAGE",), sockets["ref_image_8"])
        self.assertEqual(("IMAGE",), sockets["ref_video_2"])
        self.assertEqual(("AUDIO",), sockets["ref_video_audio_2"])
        self.assertEqual(("AUDIO",), sockets["ref_audio_2"])
        self.assertNotIn("ref_image_name", sockets)

    def test_schema_has_no_reference_count_widgets(self) -> None:
        schema = remote_minimax.RemoteMiniMaxH3ReferenceEncode.INPUT_TYPES()
        required = schema["required"]
        self.assertNotIn("ref_image_count", required)
        self.assertNotIn("ref_video_count", required)
        self.assertNotIn("ref_video_audio_count", required)
        self.assertNotIn("ref_audio_count", required)
        self.assertNotIn("peer_memory_policy", required)
        self.assertEqual("", required["clip_name"][0][0])
        self.assertEqual(["release_after_result", "keep"], schema["optional"]["peer_memory_policy"][0])
        self.assertEqual(["reuse_same_inputs", "always_refresh"], schema["optional"]["cache_policy"][0])
        self.assertEqual(("*", {"forceInput": True}), schema["optional"]["vae_name_override"])
        self.assertEqual(("*", {"forceInput": True}), schema["optional"]["audio_vae_name_override"])
        self.assertEqual({"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}, schema["hidden"])

    def test_cache_policy_uses_native_signature_by_default_and_nan_only_for_refresh(self) -> None:
        self.assertEqual(0, remote_minimax.RemoteMiniMaxH3ReferenceEncode.IS_CHANGED(cache_policy="reuse_same_inputs"))
        self.assertTrue(math.isnan(remote_minimax.RemoteMiniMaxH3ReferenceEncode.IS_CHANGED(cache_policy="always_refresh")))

    def test_validate_inputs_accepts_remote_discovered_model_aliases(self) -> None:
        result = remote_minimax.RemoteMiniMaxH3ReferenceEncode.VALIDATE_INPUTS(
            clip_name="hot-tier\\future-minimax-clip.safetensors",
            clip_type="minimax",
            device="default",
            vae_name="hot-tier\\future-video.vae",
            audio_vae_name="hot-tier\\future-audio.vae",
        )

        self.assertTrue(result)

    def test_validate_inputs_rejects_empty_loader_values(self) -> None:
        result = remote_minimax.RemoteMiniMaxH3ReferenceEncode.VALIDATE_INPUTS(
            clip_name="",
            clip_type="minimax",
            device="default",
            vae_name="video.vae",
            audio_vae_name="audio.vae",
        )

        self.assertIn("clip_name", result)

    def test_validate_inputs_rejects_invalid_cache_policy(self) -> None:
        result = remote_minimax.RemoteMiniMaxH3ReferenceEncode.VALIDATE_INPUTS(
            clip_name="minimax.safetensors",
            clip_type="minimax",
            device="default",
            vae_name="video.vae",
            audio_vae_name="audio.vae",
            cache_policy="disk_cache",
        )

        self.assertIn("cache_policy", result)

    def test_workflow_maps_images_videos_and_audio_to_exact_native_input_names(self) -> None:
        workflow = remote_minimax.build_workflow(
            clip=remote_minimax.ClipSelection("minimax.safetensors", "minimax", "default"),
            vae_name="video.vae", audio_vae_name="audio.vae", prompt="<Picture 1> <Video 1> <Audio 1>",
            width=1344, height=768, length=124, ref_image_size="match",
            image_refs=["remote/request/image.png [temp]"],
            video_refs=[[f"remote/request/video-{index}.png [temp]" for index in range(5)]],
            video_audio_refs=["remote/request/video.wav [temp]"],
            audio_refs=["remote/request/audio.wav [temp]"], request_id="request-1",
        )
        inputs = workflow["encode"]["inputs"]
        self.assertEqual(["ref_image_0", 0], inputs["ref_images.ref_image_0"])
        self.assertEqual(["ref_video_0_batch_4", 0], inputs["ref_videos.ref_video_0"])
        self.assertEqual(["ref_video_audio_0", 0], inputs["ref_video_audios.ref_video_audio_0"])
        self.assertEqual(["ref_audio_0", 0], inputs["ref_audios.ref_audio_0"])
        self.assertEqual("ImageBatch", workflow["ref_video_0_batch_4"]["class_type"])
        self.assertEqual("LoadAudio", workflow["ref_video_audio_0"]["class_type"])
        self.assertEqual("SaveRemoteLatent", workflow["save_latent"]["class_type"])
        self.assertEqual(["save", 0], workflow["save_latent"]["inputs"]["dependency"])
        self.assertEqual("VRAMCleanup", workflow["cleanup"]["class_type"])
        self.assertEqual(["save_latent", 0], workflow["cleanup"]["inputs"]["anything"])
        self.assertTrue(workflow["cleanup"]["inputs"]["offload_model"])
        self.assertTrue(workflow["cleanup"]["inputs"]["offload_cache"])
        self.assertFalse(workflow["cleanup"]["inputs"]["reset_execution_cache"])

    def test_keep_policy_does_not_add_cleanup_or_dependency_edge(self) -> None:
        workflow = remote_minimax.build_workflow(
            clip=remote_minimax.ClipSelection("minimax.safetensors", "minimax", "default"),
            vae_name="video.vae", audio_vae_name="audio.vae", prompt="scene",
            width=1344, height=768, length=124, ref_image_size="match",
            image_refs=[], video_refs=[], video_audio_refs=[], audio_refs=[],
            request_id="request-1", peer_memory_policy="keep",
        )

        self.assertNotIn("cleanup", workflow)
        self.assertNotIn("dependency", workflow["save_latent"]["inputs"])

    def test_execute_uploads_all_media_and_discards_request_inputs(self) -> None:
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
                    return {"input": {"required": {"vae_name": [["video.vae", "audio.vae"]]}}}
                return {"output": ["CONDITIONING", "LATENT"], "input": {"optional": {
                    "ref_images": [], "ref_videos": [], "ref_video_audios": [], "ref_audios": [],
                }}}

            def clip_loader_options(self):
                return {"clip_name": ["minimax.safetensors"], "clip_type": ["minimax"], "device": ["default"]}

            def upload_image(self, **kwargs):
                self.events.append(("image", kwargs["index"]))
                return f"remote-conditioning/request-1/image-{kwargs['index']}.png [temp]"

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
        with patch.object(remote_minimax, "encode_image_frame_png", return_value=b"png"), patch.object(remote_minimax, "encode_audio_wav", return_value=b"wav"):
            result = remote_minimax.execute(
                clip=remote_minimax.ClipSelection("minimax.safetensors", "minimax", "default"),
                vae_name="video.vae", audio_vae_name="audio.vae", prompt="a scene",
                width=1344, height=768, length=124, ref_image_size="max",
                images=[FakeImage(1)], videos=[FakeImage(5)], video_audios=[{"waveform": object(), "sample_rate": 44100}],
                audios=[{"waveform": object(), "sample_rate": 44100}], request_id="request-1", api=api,
                deserialize=lambda cond, latent: (cond, latent),
            )

        self.assertEqual((b"remote-conditioning/request-1/cond_00001_.cond", b"remote-conditioning/request-1/latent_00001_.latent"), result)
        self.assertEqual(("capability", "MiniMaxH3ReferenceToVideo"), api.events[0])
        self.assertEqual(("capability", "VRAMCleanup"), api.events[1])
        self.assertIn(("capability", "VAELoader"), api.events)
        self.assertEqual(5, len([event for event in api.events if event[0] == "file" and "frame" in event[1]]))
        self.assertIn(("file", "video-audio-00.wav", "audio/wav"), api.events)
        self.assertIn(("file", "audio-00.wav", "audio/wav"), api.events)
        self.assertEqual(("discard", "request-1"), api.events[-1])

    def test_missing_cleanup_capability_fails_before_upload(self) -> None:
        class FakeApi:
            def __init__(self):
                self.events = []

            def require_node(self, name):
                self.events.append(("capability", name))
                if name == "VRAMCleanup":
                    return {"input": {"required": {}, "optional": {}}, "output": ["*"]}
                return {"output": ["CONDITIONING", "LATENT"], "input": {"optional": {
                    "ref_images": [], "ref_videos": [], "ref_video_audios": [], "ref_audios": [],
                }}}

            def upload_image(self, **_kwargs):
                self.events.append("upload")
                raise AssertionError("must fail before upload")

            def submit(self, _workflow):
                self.events.append("submit")
                raise AssertionError("must fail before submit")

        api = FakeApi()
        with self.assertRaisesRegex(RuntimeError, "VRAMCleanup"):
            remote_minimax.execute(
                clip=remote_minimax.ClipSelection("minimax.safetensors", "minimax", "default"),
                vae_name="video.vae", audio_vae_name="audio.vae", prompt="a scene",
                width=1344, height=768, length=124, ref_image_size="max",
                images=[FakeImage(1)], videos=[], video_audios=[], audios=[],
                request_id="request-1", api=api, deserialize=lambda cond, latent: (cond, latent),
            )

        self.assertEqual([("capability", "MiniMaxH3ReferenceToVideo"), ("capability", "VRAMCleanup")], api.events)

    def test_connected_vae_names_use_primary_widgets_without_override_sockets(self) -> None:
        node = remote_minimax.RemoteMiniMaxH3ReferenceEncode()
        with patch.object(remote_minimax, "execute", return_value=("cond", "latent")) as execute:
            result = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name="remote-models\\video.vae",
                audio_vae_name="remote-models\\audio.vae",
                width=1344,
                height=768,
                length=124,
                ref_image_size="max",
            )

        self.assertEqual(("cond", "latent"), result)
        self.assertEqual("remote-models\\video.vae", execute.call_args.kwargs["vae_name"])
        self.assertEqual("remote-models\\audio.vae", execute.call_args.kwargs["audio_vae_name"])

    def test_reuses_same_inputs_without_resubmitting_to_peer(self) -> None:
        node = remote_minimax.RemoteMiniMaxH3ReferenceEncode()
        calls = []

        def fake_execute(**kwargs):
            calls.append(kwargs["peer_memory_policy"])
            return kwargs["deserialize"](b"cond-bytes", b"latent-bytes")

        with (
            patch.object(remote_minimax, "execute", side_effect=fake_execute),
            patch.object(remote_minimax, "decode_remote_outputs", side_effect=lambda cond, latent: (cond, {"samples": latent})),
        ):
            first = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name="video.vae",
                audio_vae_name="audio.vae",
                width=1344,
                height=768,
                length=124,
                ref_image_size="max",
                peer_memory_policy="release_after_result",
            )
            second = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name="video.vae",
                audio_vae_name="audio.vae",
                width=1344,
                height=768,
                length=124,
                ref_image_size="max",
                peer_memory_policy="keep",
            )

        self.assertEqual((b"cond-bytes", {"samples": b"latent-bytes"}), first)
        self.assertEqual(first, second)
        self.assertEqual(["release_after_result"], calls)

    def test_always_refresh_resubmits_even_with_same_inputs(self) -> None:
        node = remote_minimax.RemoteMiniMaxH3ReferenceEncode()
        calls = []

        def fake_execute(**kwargs):
            calls.append(kwargs["request_id"])
            return kwargs["deserialize"](b"cond-bytes", b"latent-bytes")

        with (
            patch.object(remote_minimax, "execute", side_effect=fake_execute),
            patch.object(remote_minimax, "decode_remote_outputs", side_effect=lambda cond, latent: (cond, {"samples": latent})),
        ):
            for _index in range(2):
                node.run(
                    prompt="scene",
                    clip_name="minimax.safetensors",
                    clip_type="minimax",
                    device="default",
                    vae_name="video.vae",
                    audio_vae_name="audio.vae",
                    width=1344,
                    height=768,
                    length=124,
                    ref_image_size="max",
                    cache_policy="always_refresh",
                )

        self.assertEqual(2, len(calls))

    def test_connected_vae_loader_objects_resolve_names_from_workflow_metadata(self) -> None:
        node = remote_minimax.RemoteMiniMaxH3ReferenceEncode()
        extra_pnginfo = {
            "workflow": {
                "nodes": [
                    {"id": 119, "type": "VAELoader", "widgets_values": ["remote-models\\video.vae"]},
                    {"id": 120, "type": "VAELoader", "widgets_values": ["remote-models\\audio.vae"]},
                    {
                        "id": 150,
                        "type": "RemoteMiniMaxH3ReferenceEncode",
                        "inputs": [
                            {"name": "vae_name", "link": 9001},
                            {"name": "audio_vae_name", "link": 9002},
                        ],
                    },
                ],
                "links": [
                    [9001, 119, 0, 150, 0, "VAE"],
                    [9002, 120, 0, 150, 1, "VAE"],
                ],
            }
        }

        with patch.object(remote_minimax, "execute", return_value=("cond", "latent")) as execute:
            result = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name=object(),
                audio_vae_name=object(),
                width=1344,
                height=768,
                length=124,
                ref_image_size="max",
                unique_id=150,
                extra_pnginfo=extra_pnginfo,
            )

        self.assertEqual(("cond", "latent"), result)
        self.assertEqual("remote-models\\video.vae", execute.call_args.kwargs["vae_name"])
        self.assertEqual("remote-models\\audio.vae", execute.call_args.kwargs["audio_vae_name"])

    def test_connected_unresolvable_vae_object_fails_with_selector_guidance(self) -> None:
        node = remote_minimax.RemoteMiniMaxH3ReferenceEncode()
        with patch.object(remote_minimax, "execute", side_effect=AssertionError("must fail before remote execution")):
            with self.assertRaisesRegex(RuntimeError, "Remote VAE Name Selector"):
                node.run(
                    prompt="scene",
                    clip_name="minimax.safetensors",
                    clip_type="minimax",
                    device="default",
                    vae_name=object(),
                    audio_vae_name="dropdown-audio.vae",
                    width=1344,
                    height=768,
                    length=124,
                    ref_image_size="max",
                    unique_id=150,
                    extra_pnginfo={"workflow": {"nodes": [], "links": []}},
                )

    def test_legacy_override_sockets_remain_backward_compatible(self) -> None:
        node = remote_minimax.RemoteMiniMaxH3ReferenceEncode()
        with patch.object(remote_minimax, "execute", return_value=("cond", "latent")) as execute:
            result = node.run(
                prompt="scene",
                clip_name="minimax.safetensors",
                clip_type="minimax",
                device="default",
                vae_name="dropdown-video.vae",
                audio_vae_name="dropdown-audio.vae",
                width=1344,
                height=768,
                length=124,
                ref_image_size="max",
                vae_name_override="remote-models\\video.vae",
                audio_vae_name_override="remote-models\\audio.vae",
            )

        self.assertEqual(("cond", "latent"), result)
        self.assertEqual("remote-models\\video.vae", execute.call_args.kwargs["vae_name"])
        self.assertEqual("remote-models\\audio.vae", execute.call_args.kwargs["audio_vae_name"])

    def test_rejects_gapped_and_short_reference_media_before_upload(self) -> None:
        with self.assertRaisesRegex(ValueError, "without gaps"):
            reference_media._connected_slots("ref_audio", {"ref_audio_1": object()}, 3)
        with self.assertRaisesRegex(ValueError, "at least 5 frames"):
            remote_minimax._ordered_images("ref_video", {"ref_video_0": FakeImage(4)}, batches=True, maximum=3)

    def test_rejects_the_legacy_flat_peer_reference_contract_before_upload(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "nested reference input contract"):
            remote_minimax.validate_ref2va_capability({"output": ["CONDITIONING", "LATENT"], "input": {"optional": {}}})

    def test_timeout_and_caller_interrupt_cancel_the_submitted_ref2va_prompt(self) -> None:
        class CallerInterrupted(BaseException):
            pass

        class Api:
            def __init__(self, failure):
                self.failure = failure
                self.cancelled = []

            def require_node(self, name):
                if name == "MiniMaxH3ReferenceToVideo":
                    return {"output": ["CONDITIONING", "LATENT"], "input": {"optional": {
                        "ref_images": [], "ref_videos": [], "ref_video_audios": [], "ref_audios": [],
                    }}}
                if name == "VAELoader":
                    return {"input": {"required": {"vae_name": [["video.vae", "audio.vae"]]}}}
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
                remote_minimax.execute(
                    clip=remote_minimax.ClipSelection("minimax.safetensors", "minimax", "default"),
                    vae_name="video.vae", audio_vae_name="audio.vae", prompt="scene",
                    width=1344, height=768, length=124, ref_image_size="max",
                    images=[], videos=[], video_audios=[], audios=[], request_id="request-1", api=api,
                    deserialize=lambda cond, latent: (cond, latent), peer_memory_policy="keep",
                )
            self.assertEqual(["prompt-1"], api.cancelled)


if __name__ == "__main__":
    unittest.main()

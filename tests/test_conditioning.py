from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import torch

import remoteworker.conditioning.service as execution_remote_conditioning
from remoteworker.core.media import DynamicImageInputs


class execution_FakeComfyApi:
    def __init__(self) -> None:
        self.workflow = None
        self.artifact = None
        self.events = []

    def submit(self, workflow):
        self.events.append("submit")
        self.workflow = workflow
        return "prompt-99"

    def require_node(self, node_name):
        self.events.append(("capability", node_name))
        if node_name == "VRAMCleanup":
            return {
                "input": {
                    "required": {
                        "offload_model": ["BOOLEAN"],
                        "offload_cache": ["BOOLEAN"],
                        "reset_execution_cache": ["BOOLEAN"],
                    },
                    "optional": {"anything": ["*"]},
                },
                "output": ["*"],
            }
        raise AssertionError(node_name)

    def wait_for_history(self, _prompt_id):
        return {
            "prompt-99": {
                "status": {"status_str": "success"},
                "outputs": {"save": {"text": ["remote-conditioning/request-42/cond_00001_.cond"]}},
            }
        }

    def download_artifact(self, artifact):
        self.artifact = artifact
        return b"trusted-cond"


class RemoteExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        execution_remote_conditioning.clear_conditioning_cache()

    def test_executes_the_user_selected_clip_and_request_scoped_artifact(self) -> None:
        api = execution_FakeComfyApi()
        clip = execution_remote_conditioning.ClipSelection("other-clip.safetensors", "flux2", "default")
        result = execution_remote_conditioning.execute(
            clip=clip,
            prompt="a glass bird",
            request_id="request-42",
            api=api,
            deserialize=lambda data: {"loaded": data},
        )

        self.assertEqual({"loaded": b"trusted-cond"}, result)
        self.assertEqual("other-clip.safetensors", api.workflow["clip"]["inputs"]["clip_name"])
        self.assertEqual("flux2", api.workflow["clip"]["inputs"]["type"])
        self.assertEqual("remote-conditioning/request-42/cond_00001_.cond", api.artifact)
        self.assertEqual([("capability", "VRAMCleanup"), "submit"], api.events)

    def test_keep_policy_does_not_probe_or_add_peer_cleanup(self) -> None:
        api = execution_FakeComfyApi()
        result = execution_remote_conditioning.execute(
            clip=execution_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
            prompt="a glass bird",
            request_id="request-42",
            api=api,
            deserialize=lambda data: data,
            peer_memory_policy="keep",
        )

        self.assertEqual(b"trusted-cond", result)
        self.assertEqual(["submit"], api.events)
        self.assertNotIn("cleanup", api.workflow)

    def test_release_cleanup_capability_fails_before_text_submit(self) -> None:
        class Api(execution_FakeComfyApi):
            def require_node(self, node_name):
                self.events.append(("capability", node_name))
                return {"input": {"required": {}, "optional": {}}, "output": ["*"]}

        api = Api()
        with self.assertRaisesRegex(RuntimeError, "VRAMCleanup"):
            execution_remote_conditioning.execute(
                clip=execution_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
                prompt="a glass bird",
                request_id="request-42",
                api=api,
                deserialize=lambda data: data,
            )

        self.assertEqual([("capability", "VRAMCleanup")], api.events)

    def test_timeout_and_caller_interrupt_cancel_the_submitted_text_prompt(
        self,
    ) -> None:
        class CallerInterrupted(BaseException):
            pass

        class Api(execution_FakeComfyApi):
            def __init__(self, failure):
                super().__init__()
                self.failure = failure

            def wait_for_history(self, _prompt_id):
                raise self.failure

            def cancel_prompt(self, prompt_id):
                self.events.append(("cancel", prompt_id))

        for failure in (TimeoutError("peer wait expired"), CallerInterrupted()):
            api = Api(failure)
            with self.assertRaises(type(failure)):
                execution_remote_conditioning.execute(
                    clip=execution_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
                    prompt="a glass bird",
                    request_id="request-42",
                    api=api,
                    deserialize=lambda data: data,
                )
            self.assertIn(("cancel", "prompt-99"), api.events)

    def test_cancel_failure_warns_and_preserves_the_original_timeout(self) -> None:
        class Api(execution_FakeComfyApi):
            def wait_for_history(self, _prompt_id):
                raise TimeoutError("peer wait expired")

            def cancel_prompt(self, _prompt_id):
                raise RuntimeError("cancel endpoint unavailable")

        with self.assertLogs("remoteworker.core.execution", level="WARNING") as caught:
            with self.assertRaisesRegex(TimeoutError, "peer wait expired"):
                execution_remote_conditioning.execute(
                    clip=execution_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
                    prompt="a glass bird",
                    request_id="request-42",
                    api=Api(),
                    deserialize=lambda data: data,
                )

        self.assertTrue(any("could not cancel remote prompt prompt-99" in message for message in caught.output))

    def test_master_node_reuses_same_input_conditioning_without_peer_resubmit(
        self,
    ) -> None:
        stream = io.BytesIO()
        torch.save(["cached-cond"], stream)
        calls = []

        def execute_once(**_kwargs):
            calls.append("remote")
            return stream.getvalue()

        node = execution_remote_conditioning.RemoteConditioning()
        kwargs = dict(
            prompt="a glass bird",
            conditioning_strategy="text",
            remote_text_encoder="example-vision-encoder.safetensors",
            clip_type="krea2",
            device="default",
            image_count=1,
            grounding_px=768,
            system_prompt="",
            server_url="http://127.0.0.1:8188",
            timeout_seconds=120,
            peer_memory_policy="release_after_result",
            cache_policy="reuse_same_inputs",
        )

        with patch.object(
            execution_remote_conditioning,
            "execute_strategy_bytes",
            side_effect=execute_once,
        ):
            first = node.run(**kwargs)
            second = node.run(**kwargs)

        self.assertEqual((["cached-cond"],), first)
        self.assertEqual((["cached-cond"],), second)
        self.assertEqual(["remote"], calls)

    def test_always_refresh_bypasses_process_local_conditioning_cache(self) -> None:
        stream = io.BytesIO()
        torch.save(["fresh-cond"], stream)

        node = execution_remote_conditioning.RemoteConditioning()
        kwargs = dict(
            prompt="a glass bird",
            conditioning_strategy="text",
            remote_text_encoder="example-vision-encoder.safetensors",
            clip_type="krea2",
            device="default",
            image_count=1,
            grounding_px=768,
            system_prompt="",
            server_url="http://127.0.0.1:8188",
            timeout_seconds=120,
            peer_memory_policy="release_after_result",
            cache_policy="always_refresh",
        )

        with patch.object(
            execution_remote_conditioning,
            "execute_strategy_bytes",
            return_value=stream.getvalue(),
        ) as execute_bytes:
            node.run(**kwargs)
            node.run(**kwargs)

        self.assertEqual(2, execute_bytes.call_count)


import remoteworker.conditioning.service as image_remote_conditioning


class image_FakeBatch:
    def __init__(self, count: int, height: int = 64, width: int = 96, channels: int = 3):
        self.shape = (count, height, width, channels)

    def __getitem__(self, index):
        return (self, index)


class image_FakeFrame:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def to(self, **_kwargs):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.array


class OrderedImageAttachmentTests(unittest.TestCase):
    def test_dynamic_image_inputs_accept_only_numbered_image_slots(self) -> None:
        inputs = DynamicImageInputs()
        self.assertIn("image_37", inputs)
        self.assertEqual(("IMAGE",), inputs["image_37"])
        self.assertNotIn("image_b", inputs)

    def test_socket_and_batch_order_are_preserved_without_shape_merging(self) -> None:
        first = image_FakeBatch(2, 64, 96)
        second = image_FakeBatch(1, 720, 480)
        frames = image_remote_conditioning.ordered_image_frames(2, image_1=first, image_2=second)
        self.assertEqual([(first, 0), (first, 1), (second, 0)], frames)

    def test_krea_strategy_rejects_more_than_two_images_before_workflow_build(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "krea2-grounded.*1 to 2"):
            image_remote_conditioning.build_strategy_workflow(
                strategy_name="krea2-grounded",
                clip=image_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
                prompt="edit the subject",
                request_id="request-9",
                image_refs=["a.png [temp]", "b.png [temp]", "c.png [temp]"],
                grounding_px=768,
                system_prompt="",
            )

    def test_krea_strategy_maps_two_ordered_images_without_changing_transport(
        self,
    ) -> None:
        workflow = image_remote_conditioning.build_strategy_workflow(
            strategy_name="krea2-grounded",
            clip=image_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
            prompt="move the subject",
            request_id="request-9",
            image_refs=[
                "request-9/image-0001.png [temp]",
                "request-9/image-0002.png [temp]",
            ],
            grounding_px=1024,
            system_prompt="identity detail",
        )
        self.assertEqual("Krea2EditGroundedEncode", workflow["encode"]["class_type"])
        self.assertEqual(["image_1", 0], workflow["encode"]["inputs"]["image"])
        self.assertEqual(["image_2", 0], workflow["encode"]["inputs"]["image_b"])
        self.assertEqual(1024, workflow["encode"]["inputs"]["grounding_px"])
        self.assertEqual("identity detail", workflow["encode"]["inputs"]["system_prompt"])
        self.assertEqual(["save", 0], workflow["cleanup"]["inputs"]["anything"])
        self.assertFalse(workflow["cleanup"]["inputs"]["reset_execution_cache"])

    def test_krea_keep_policy_does_not_add_cleanup(self) -> None:
        workflow = image_remote_conditioning.build_strategy_workflow(
            strategy_name="krea2-grounded",
            clip=image_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
            prompt="move the subject",
            request_id="request-9",
            image_refs=["request-9/image-0001.png [temp]"],
            grounding_px=768,
            system_prompt="",
            peer_memory_policy="keep",
        )

        self.assertNotIn("cleanup", workflow)

    def test_master_schema_exposes_strategy_and_dynamic_image_count(self) -> None:
        schema = image_remote_conditioning.RemoteConditioning.INPUT_TYPES()
        self.assertIn("conditioning_strategy", schema["required"])
        self.assertIn("image_count", schema["required"])
        self.assertEqual(schema["required"]["image_count"][1]["default"], 1)
        self.assertEqual(schema["required"]["image_count"][1]["min"], 1)
        self.assertIn("image_99", schema["optional"])

    def test_png_encoding_preserves_each_frame_dimensions(self) -> None:
        import numpy as np
        from PIL import Image

        encoded = image_remote_conditioning.encode_image_frame_png(
            image_FakeFrame(np.zeros((37, 53, 3), dtype=np.float32))
        )
        with Image.open(io.BytesIO(encoded)) as image:
            self.assertEqual((53, 37), image.size)

    def test_strategy_checks_capability_then_uploads_in_order(self) -> None:
        class FakeApi:
            def __init__(self):
                self.events = []

            def require_node(self, name):
                self.events.append(("capability", name))
                if name == "VRAMCleanup":
                    return {
                        "input": {
                            "required": {
                                "offload_model": ["BOOLEAN"],
                                "offload_cache": ["BOOLEAN"],
                                "reset_execution_cache": ["BOOLEAN"],
                            },
                            "optional": {"anything": ["*"]},
                        },
                        "output": ["*"],
                    }
                return {
                    "input": {
                        "required": {"clip": ("CLIP",), "prompt": ("STRING",)},
                        "optional": {"image": ("IMAGE",), "image_b": ("IMAGE",)},
                    },
                    "output": ["CONDITIONING"],
                }

            def upload_image(self, *, request_id, index, png):
                self.events.append(("upload", index, png))
                return f"remote-conditioning/{request_id}/image-{index:04d}.png [temp]"

            def submit(self, workflow):
                self.events.append(("submit", workflow))
                return "prompt-9"

            def wait_for_history(self, _prompt_id):
                return {
                    "prompt-9": {
                        "status": {"status_str": "success"},
                        "outputs": {"save": {"text": ["remote-conditioning/request-9/cond_00001_.cond"]}},
                    }
                }

            def download_artifact(self, _artifact):
                return b"cond"

            def discard_inputs(self, request_id):
                self.events.append(("discard", request_id))
                return True

        api = FakeApi()
        with patch.object(
            image_remote_conditioning,
            "encode_image_frame_png",
            side_effect=[b"one", b"two"],
        ):
            result = image_remote_conditioning.execute_strategy(
                strategy_name="krea2-grounded",
                clip=image_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
                prompt="edit",
                request_id="request-9",
                frames=[object(), object()],
                grounding_px=768,
                system_prompt="",
                api=api,
                deserialize=lambda data: data,
            )

        self.assertEqual(b"cond", result)
        self.assertEqual(("capability", "Krea2EditGroundedEncode"), api.events[0])
        self.assertEqual(("capability", "VRAMCleanup"), api.events[1])
        self.assertEqual([1, 2], [event[1] for event in api.events if event[0] == "upload"])
        self.assertEqual(("discard", "request-9"), api.events[-1])

    def test_release_cleanup_capability_fails_before_grounded_upload(self) -> None:
        class Api:
            def __init__(self):
                self.events = []

            def require_node(self, name):
                self.events.append(("capability", name))
                if name == "Krea2EditGroundedEncode":
                    return {
                        "input": {
                            "required": {"clip": ["CLIP"], "prompt": ["STRING"]},
                            "optional": {"image": ["IMAGE"]},
                        },
                        "output": ["CONDITIONING"],
                    }
                return {"input": {"required": {}, "optional": {}}, "output": ["*"]}

            def upload_image(self, **_kwargs):
                self.events.append("upload")
                raise AssertionError("upload must not happen before cleanup capability validation")

        api = Api()
        with self.assertRaisesRegex(RuntimeError, "VRAMCleanup"):
            image_remote_conditioning.execute_strategy(
                strategy_name="krea2-grounded",
                clip=image_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
                prompt="edit",
                request_id="request-9",
                frames=[object()],
                grounding_px=768,
                system_prompt="",
                api=api,
                deserialize=lambda data: data,
            )

        self.assertEqual(
            [("capability", "Krea2EditGroundedEncode"), ("capability", "VRAMCleanup")],
            api.events,
        )

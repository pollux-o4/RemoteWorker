"""Node-owned discovery through planning, speculative execution and consumption."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from remoteworker.preparation import integration, plan
from remoteworker.preparation.capabilities import Preparation, discover, registered
from remoteworker.preparation.coordinator import Coordinator
from remoteworker.preparation.handoff import Handoffs
from remoteworker.qwen.service import RemoteQwenImage21Service, preparation_inputs


class CapabilityContracts(unittest.TestCase):
    def test_new_node_roles_drive_plan_without_scheduler_edits(self):
        class FutureClip:
            REMOTE_PREPARATION = Preparation("encoder")

        class FutureVae:
            REMOTE_PREPARATION = Preparation("decoder", kind="audio")

        classes = {"FutureClip": FutureClip, "FutureVae": FutureVae, "Other": object}
        graph = {
            "clip": {
                "class_type": "FutureClip",
                "inputs": {"server_url": "http://peer"},
            },
            "vae": {
                "class_type": "FutureVae",
                "inputs": {"server_url": "http://peer", "vae_name": "audio"},
            },
        }
        with patch(
            "remoteworker.preparation.capabilities.registered",
            return_value=discover(classes),
        ):
            self.assertEqual([x[0] for x in plan.next_encoders(graph)], ["clip"])
            self.assertEqual(plan.first_decoder(graph), ("vae", "http://peer", "audio", "audio"))

    def test_qwen_is_discovered_and_autogrow_is_adapted(self):
        spec = registered()["RemoteQwenImage21"][1]
        self.assertEqual(spec.role, "encoder")
        self.assertEqual(
            spec.inputs({"images.image_1": "image", "prompt": "p"}),
            {"image_1": "image", "prompt": "p"},
        )

    def test_qwen_prefetch_and_foreground_consume_once_with_refresh(self):
        values = dict(
            prompt="p",
            negative_prompt="",
            server_url="http://peer:18293",
            timeout_seconds=120,
            remote_text_encoder="clip",
            vae_name="vae",
            resolution=512,
            negative_vision=False,
            cache_policy="always_refresh",
            **{"images.image_1": "frame"},
        )
        graph = {"qwen": {"class_type": "RemoteQwenImage21", "inputs": values}}
        current = (0, "current", {}, {}, [])
        following = (1, "next", graph, {}, ["qwen"])
        server = SimpleNamespace(
            port=18292,
            prompt_queue=SimpleNamespace(get_current_queue=lambda: ([current], [following])),
        )
        owner = Coordinator(
            server,
            {"RemoteQwenImage21": RemoteQwenImage21Service},
            lambda *a: None,
            sampling=lambda *a: True,
            resolve=lambda g, k: dict(g[k]["inputs"]),
            handoffs=Handoffs(),
        )
        original = RemoteQwenImage21Service.run
        with (
            patch.object(
                RemoteQwenImage21Service,
                "run",
                integration.managed(original, "encoder"),
            ),
            patch.object(integration, "coordinator", owner),
            patch.object(integration, "current_prompt", return_value="next"),
            patch("remoteworker.qwen.service.execute", return_value=(b"p", b"n", b"l")) as execute,
            patch(
                "remoteworker.qwen.service._decode",
                return_value=("positive", "negative", {"samples": "latent"}),
            ),
            patch("remoteworker.qwen.service._ordered_qwen_image_frames", return_value=[]),
        ):
            owner.tick()
            self.assertEqual(owner.events[-1]["state"], "conditioning_ready")
            self.assertGreater(owner.handoffs.bytes_used, 0)
            result = RemoteQwenImage21Service().run(**preparation_inputs(values), unique_id="qwen")
            self.assertEqual(result[0], "positive")
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(owner.handoffs.bytes_used, 0)
            RemoteQwenImage21Service().run(**preparation_inputs(values), unique_id="qwen")
            self.assertEqual(execute.call_count, 2)

    def test_vae_only_graph_warms_without_encoder(self):
        graph = {
            "v": {
                "class_type": "RemoteVAEDecode",
                "inputs": {"server_url": "http://peer", "vae_name": "vae"},
            }
        }
        current = (0, "current", graph)
        server = SimpleNamespace(
            port=18292,
            prompt_queue=SimpleNamespace(get_current_queue=lambda: ([current], [])),
        )
        calls = []
        owner = Coordinator(server, {}, lambda *args: calls.append(args), sampling=lambda *a: True)
        owner.tick()
        self.assertEqual(calls, [("http://peer", "vae", "image")])

    @unittest.skipUnless(
        __import__("remoteworker.qwen.service", fromlist=["comfy_io"]).comfy_io,
        "Native Comfy V3 unavailable",
    )
    def test_native_v3_resize_ancestor_and_hidden_node_identity(self):
        import torch
        from comfy_extras.nodes_post_processing import ResizeImageMaskNode

        from remoteworker.preparation.inputs import Resolver
        from remoteworker.qwen import service as qwen

        class Image:
            FUNCTION = "run"

            def run(self):
                return (torch.zeros(1, 16, 24, 3),)

        graph = {
            "image": {"class_type": "LoadImage", "inputs": {}},
            "resize": {
                "class_type": "ResizeImageMaskNode",
                "inputs": {
                    "input": ["image", 0],
                    "scale_method": "area",
                    "resize_type": "scale by",
                    "resize_type.multiplier": 2.0,
                },
            },
            "qwen": {"inputs": {"images.image_1": ["resize", 0]}},
        }
        # Use the native option value rather than duplicating its spelling.
        from comfy_extras.nodes_post_processing import ResizeType

        graph["resize"]["inputs"]["resize_type"] = ResizeType.SCALE_BY
        values = Resolver(graph, {"LoadImage": Image, "ResizeImageMaskNode": ResizeImageMaskNode}).inputs("qwen")
        self.assertEqual(tuple(values["images.image_1"].shape), (1, 32, 48, 3))
        self.assertIn(
            qwen.comfy_io.Hidden.unique_id,
            qwen.RemoteQwenImage21Node.define_schema().hidden,
        )
        args = dict(
            server_url="http://peer",
            timeout_seconds=120,
            remote_text_encoder="clip",
            vae_name="vae",
            prompt="p",
            negative_prompt="",
            resolution=512,
            negative_vision=False,
            peer_memory_policy="release_after_result",
            cache_policy="always_refresh",
        )
        with (
            patch.object(qwen.RemoteQwenImage21Node, "hidden", SimpleNamespace(unique_id="qwen")),
            patch.object(qwen.RemoteQwenImage21Service, "run", return_value=("p", "n", {})) as run,
        ):
            qwen.RemoteQwenImage21Node.execute(**args, images={"image_1": values["images.image_1"]})
            self.assertEqual(run.call_args.kwargs["unique_id"], "qwen")
            self.assertEqual(tuple(run.call_args.kwargs["image_1"].shape), (1, 32, 48, 3))

"""Picker visibility and localized labels must preserve node wire contracts."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from remoteworker import _native, registry
from remoteworker.prompt import service as prompt
from remoteworker.qwen import worker as qwen
from remoteworker.vae import service as vae


class InternalWorkerMetadata(unittest.TestCase):
    def test_hidden_workers_keep_registered_ids_and_callable_execution_contracts(self):
        for name, node, output_types in (
            ("RemotePromptGenerate", prompt.RemotePromptGenerate, ("STRING",)),
            (
                "RemoteQwenImage21Encode",
                qwen.RemoteQwenImage21Encode,
                ("CONDITIONING", "CONDITIONING", "LATENT"),
            ),
        ):
            with self.subTest(node=name):
                self.assertTrue(node.DEV_ONLY)
                self.assertIs(registry.NODE_CLASS_MAPPINGS[name], node)
                self.assertTrue(registry.NODE_DISPLAY_NAME_MAPPINGS[name])
                self.assertEqual(node.RETURN_TYPES, output_types)
                self.assertTrue(callable(getattr(node(), node.FUNCTION)))

    def test_hidden_prompt_worker_dispatches_registered_inputs_to_native_execution(self):
        clip, image, video = object(), object(), object()
        expected = ("generated prompt",)
        with patch.object(_native, "prompt_generate_native", return_value=expected) as run:
            node = registry.NODE_CLASS_MAPPINGS["RemotePromptGenerate"]()
            result = getattr(node, node.FUNCTION)(
                clip=clip,
                prompt="scene",
                system_prompt="Describe the scene.",
                media_strategy="minimax-ref2va",
                max_length=256,
                image=image,
                video_0=video,
            )
        self.assertIs(result, expected)
        run.assert_called_once_with(
            prompt,
            clip,
            "scene",
            "Describe the scene.",
            "minimax-ref2va",
            256,
            image,
            {"video_0": video},
        )

    def test_hidden_qwen_worker_preserves_optional_vision_and_latent_dispatch(self):
        clip, model, image, latent = object(), object(), object(), object()
        expected = (["positive"], ["negative"], {"samples": latent})
        with patch.object(_native, "qwen_encode_native", return_value=expected) as encode:
            node = registry.NODE_CLASS_MAPPINGS["RemoteQwenImage21Encode"]()
            result = getattr(node, node.FUNCTION)(
                clip=clip,
                prompt="edit",
                negative_prompt="blur",
                resolution=512,
                vae=model,
                negative_vision=True,
                image_1=image,
            )
        self.assertIs(result, expected)
        encode.assert_called_once_with(
            clip, "edit", "blur", 512, model, True, {"image_1": image}
        )


class LocalizedVaeTransferMetadata(unittest.TestCase):
    def test_transfer_labels_describe_tensor_results_and_inactive_legacy_output(self):
        root = Path(__file__).resolve().parents[1]
        for language, tensor_claim, shared_label, direct_label, unused, no_output in (
            ("ko", r"두.*텐서", "공유", "직접", "미사용", "실행 결과 없음"),
            ("en", r"both.*tensors?", "shared", "http", "unused", "no output"),
        ):
            definitions = json.loads(
                (root / "locales" / language / "nodeDefs.json").read_text()
            )
            for node, primary_type in (
                (vae.RemoteVAEDecode, "IMAGE"),
                (vae.RemoteVAEDecodeAudio, "AUDIO"),
            ):
                with self.subTest(language=language, node=node.__name__):
                    localized = definitions[node.__name__]
                    transfer = localized["inputs"]["output_mode"]
                    wire_modes = node.INPUT_TYPES()["optional"]["output_mode"][0]
                    self.assertEqual(set(transfer["options"]), set(wire_modes))
                    self.assertIn(shared_label, transfer["options"]["shared"].lower())
                    self.assertIn(direct_label, transfer["options"]["tensor"].lower())
                    self.assertRegex(transfer["tooltip"].lower(), tensor_claim)
                    self.assertRegex(transfer["tooltip"], r"512\s*MiB")
                    self.assertEqual((primary_type, "STRING"), node.RETURN_TYPES)
                    legacy = localized["outputs"][1]
                    self.assertIn(unused, legacy["name"].lower())
                    self.assertIn(no_output, legacy["tooltip"].lower())
                    self.assertIn("inactive", node.OUTPUT_TOOLTIPS[1].lower())

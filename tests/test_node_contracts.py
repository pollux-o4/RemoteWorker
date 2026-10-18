from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import unittest
from pathlib import Path

from remoteworker.conditioning.workflow import extract_artifact_path
from remoteworker.core.protocol import build_release_cleanup_node

REGISTRATION_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

REGISTRATION_SPEC = importlib.util.spec_from_file_location(
    "custom_nodes_under_test",
    REGISTRATION_PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(REGISTRATION_PACKAGE_ROOT)],
)

assert REGISTRATION_SPEC and REGISTRATION_SPEC.loader

registration_custom_nodes = importlib.util.module_from_spec(REGISTRATION_SPEC)

sys.modules[REGISTRATION_SPEC.name] = registration_custom_nodes

REGISTRATION_SPEC.loader.exec_module(registration_custom_nodes)


class NodeRegistrationTests(unittest.TestCase):
    def test_registry_rejects_duplicate_node_ids(self):
        from types import SimpleNamespace

        from remoteworker.registry import collect_nodes
        module = SimpleNamespace(__name__="fixture", NODE_CLASS_MAPPINGS={"Shared": object}, NODE_DISPLAY_NAME_MAPPINGS={"Shared": "Shared"})
        with self.assertRaisesRegex(ValueError, "Duplicate ComfyUI node IDs"):
            collect_nodes((module, module))

    def test_registry_rejects_missing_display_names(self):
        from types import SimpleNamespace

        from remoteworker.registry import collect_nodes
        module = SimpleNamespace(__name__="fixture", NODE_CLASS_MAPPINGS={"Missing": object}, NODE_DISPLAY_NAME_MAPPINGS={})
        with self.assertRaisesRegex(ValueError, "display names do not match"):
            collect_nodes((module,))

    def test_remote_conditioning_master_is_registered_with_existing_nodes(self) -> None:
        self.assertIn("SaveConditioning", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("SaveRemoteConditioning", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("SaveRemoteLatent", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("SaveRemoteText", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("LoadConditioning", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("RemoteConditioning", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn(
            "RemoteMiniMaxH3ReferenceEncode",
            registration_custom_nodes.NODE_CLASS_MAPPINGS,
        )
        self.assertIn("RemoteMiniMaxH3ImageToVideo", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("RemotePromptEnhancer", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("RemoteVAESelector", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("RemoteQwenImage21Encode", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("RemoteQwenImage21", registration_custom_nodes.NODE_CLASS_MAPPINGS)
        self.assertNotIn(
            "RemotePromptEnhancerProfileManager",
            registration_custom_nodes.NODE_CLASS_MAPPINGS,
        )


import remoteworker.conditioning.service as peer_remote_conditioning


class ServerUrlTests(unittest.TestCase):
    def test_accepts_localhost_port_and_hosted_https_origin(self) -> None:
        self.assertEqual(
            "http://127.0.0.1:8188",
            peer_remote_conditioning._validate_server_url("http://127.0.0.1:8188"),
        )
        self.assertEqual(
            "https://comfy.example",
            peer_remote_conditioning._validate_server_url("https://comfy.example/"),
        )

    def test_rejects_non_origin_or_unsupported_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "server_url"):
            peer_remote_conditioning._validate_server_url("localhost:8188")
        with self.assertRaisesRegex(ValueError, "server_url"):
            peer_remote_conditioning._validate_server_url("https://comfy.example/api/prompt")


import remoteworker.integration.vae_selector as vae_remote_vae_selector


class RemoteVAESelectorContractTests(unittest.TestCase):
    def test_selector_returns_only_the_remote_model_name_string(self) -> None:
        result = vae_remote_vae_selector.RemoteVAESelector().run(
            server_url="http://127.0.0.1:8188",
            timeout_seconds=30,
            vae_name="remote-models\\minimax_h3_video_vae_fp16.safetensors",
        )

        self.assertEqual(("remote-models\\minimax_h3_video_vae_fp16.safetensors",), result)

    def test_schema_has_no_local_vae_object_socket(self) -> None:
        schema = vae_remote_vae_selector.RemoteVAESelector.INPUT_TYPES()

        self.assertEqual(("STRING",), vae_remote_vae_selector.RemoteVAESelector.RETURN_TYPES)
        self.assertEqual(("vae_name",), vae_remote_vae_selector.RemoteVAESelector.RETURN_NAMES)
        self.assertIn("server_url", schema["required"])
        self.assertIn("vae_name", schema["required"])
        self.assertNotIn("VAE", repr(schema))


import remoteworker.conditioning.service as workflow_remote_conditioning


class ClipSelectionTemplateTests(unittest.TestCase):
    def test_remote_clip_combo_values_use_node_validation_instead_of_static_membership(
        self,
    ) -> None:
        validator = workflow_remote_conditioning.RemoteConditioning.VALIDATE_INPUTS
        parameters = inspect.signature(validator).parameters
        self.assertTrue({"remote_text_encoder", "clip_type", "device"}.issubset(parameters))
        self.assertTrue(
            validator(
                "example-text-encoder.safetensors",
                "lumina2",
                "default",
            )
        )
        self.assertIn("remote_text_encoder", validator("", "lumina2", "default"))

    def test_selected_clip_loader_values_build_one_prompt_to_conditioning_workflow(
        self,
    ) -> None:
        clip = workflow_remote_conditioning.ClipSelection("custom-clip.safetensors", "lumina2", "cpu")
        workflow = workflow_remote_conditioning.build_workflow(clip, "a glass bird", "request-42")

        self.assertEqual("custom-clip.safetensors", workflow["clip"]["inputs"]["clip_name"])
        self.assertEqual("lumina2", workflow["clip"]["inputs"]["type"])
        self.assertEqual("cpu", workflow["clip"]["inputs"]["device"])
        self.assertEqual("a glass bird", workflow["encode"]["inputs"]["text"])
        self.assertEqual(["encode", 0], workflow["save"]["inputs"]["conditioning"])
        self.assertEqual(
            "remote-conditioning/request-42/cond",
            workflow["save"]["inputs"]["filename_prefix"],
        )
        self.assertEqual("VRAMCleanup", workflow["cleanup"]["class_type"])
        self.assertEqual(["save", 0], workflow["cleanup"]["inputs"]["anything"])
        self.assertTrue(workflow["cleanup"]["inputs"]["offload_model"])
        self.assertTrue(workflow["cleanup"]["inputs"]["offload_cache"])
        self.assertFalse(workflow["cleanup"]["inputs"]["reset_execution_cache"])

    def test_release_cleanup_payload_is_centralized(self) -> None:
        cleanup = build_release_cleanup_node(["save", 0])

        self.assertEqual("VRAMCleanup", cleanup["class_type"])
        self.assertEqual(
            {
                "anything": ["save", 0],
                "offload_model": True,
                "offload_cache": True,
                "reset_execution_cache": False,
            },
            cleanup["inputs"],
        )

    def test_keep_policy_preserves_the_existing_workflow_without_cleanup(self) -> None:
        workflow = workflow_remote_conditioning.build_workflow(
            workflow_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"),
            "a glass bird",
            "request-42",
            peer_memory_policy="keep",
        )

        self.assertNotIn("cleanup", workflow)

    def test_request_id_and_clip_selection_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "request_id"):
            workflow_remote_conditioning.build_workflow(
                workflow_remote_conditioning.ClipSelection("text-encoder.safetensors", "lumina2", "default"), "text", "../../escape"
            )
        with self.assertRaisesRegex(ValueError, "CLIP selection"):
            workflow_remote_conditioning.build_workflow(
                workflow_remote_conditioning.ClipSelection("", "lumina2", "default"),
                "text",
                "request-42",
            )

    def test_history_requires_exactly_one_artifact_in_the_request_directory(
        self,
    ) -> None:
        history = {
            "prompt-99": {
                "status": {"status_str": "success"},
                "outputs": {"save": {"text": ["remote-conditioning/request-42/cond_00001_.cond"]}},
            }
        }
        self.assertEqual(
            "remote-conditioning/request-42/cond_00001_.cond",
            extract_artifact_path(history, "prompt-99", "request-42"),
        )

    def test_remote_execution_error_surfaces_node_and_exception_details(self) -> None:
        history = {
            "prompt-99": {
                "status": {
                    "status_str": "error",
                    "messages": [
                        [
                            "execution_error",
                            {
                                "node_type": "CLIPLoader",
                                "exception_type": "KeyError",
                                "exception_message": "'int8_tensorwise'",
                            },
                        ]
                    ],
                },
                "outputs": {},
            }
        }
        with self.assertRaisesRegex(RuntimeError, r"CLIPLoader.*KeyError.*int8_tensorwise"):
            extract_artifact_path(history, "prompt-99", "request-42")

    def test_master_node_has_embedded_prompt_clip_loader_selection_and_conditioning_output(
        self,
    ) -> None:
        schema = workflow_remote_conditioning.RemoteConditioning.INPUT_TYPES()
        self.assertTrue(
            {
                "prompt",
                "remote_text_encoder",
                "clip_type",
                "device",
                "server_url",
            }.issubset(schema["required"])
        )
        self.assertIn("peer_memory_policy", schema["optional"])
        self.assertIn("cache_policy", schema["optional"])
        for field, default in zip(
            ("remote_text_encoder", "clip_type", "device"),
            workflow_remote_conditioning.ZIT_DEFAULT_CLIP,
        ):
            self.assertEqual([default], schema["required"][field][0])
            self.assertEqual(default, schema["required"][field][1]["default"])
        self.assertEqual("http://127.0.0.1:8188", schema["required"]["server_url"][1]["default"])
        self.assertEqual(
            ["release_after_result", "keep"],
            schema["optional"]["peer_memory_policy"][0],
        )
        self.assertEqual(
            "release_after_result",
            schema["optional"]["peer_memory_policy"][1]["default"],
        )
        self.assertEqual(
            ["reuse_same_inputs", "always_refresh"],
            schema["optional"]["cache_policy"][0],
        )
        self.assertEqual("reuse_same_inputs", schema["optional"]["cache_policy"][1]["default"])
        self.assertEqual(
            ("CONDITIONING",),
            workflow_remote_conditioning.RemoteConditioning.RETURN_TYPES,
        )

    def test_cache_policy_uses_native_signature_by_default_and_nan_only_for_refresh(
        self,
    ) -> None:
        self.assertEqual(
            0,
            workflow_remote_conditioning.RemoteConditioning.IS_CHANGED(cache_policy="reuse_same_inputs"),
        )
        self.assertTrue(
            math.isnan(workflow_remote_conditioning.RemoteConditioning.IS_CHANGED(cache_policy="always_refresh"))
        )

    def test_invalid_policies_fail_fast_at_input_validation(self) -> None:
        self.assertIn(
            "peer_memory_policy",
            workflow_remote_conditioning.RemoteConditioning.VALIDATE_INPUTS(
                "clip.safetensors",
                "lumina2",
                "default",
                peer_memory_policy="drop_everything",
            ),
        )
        self.assertIn(
            "cache_policy",
            workflow_remote_conditioning.RemoteConditioning.VALIDATE_INPUTS(
                "clip.safetensors", "lumina2", "default", cache_policy="disk_cache"
            ),
        )


class LocalDefaultBoundaryTests(unittest.TestCase):
    def _probe(self, settings_path, module='remote_defaults'):
        import os
        import subprocess
        env = dict(os.environ, REMOTEWORKER_CONFIG=settings_path)
        result = subprocess.run([sys.executable, '-c',
            'import json; import remoteworker.defaults as remote_defaults; print(json.dumps(remote_defaults.LOCAL_SETTINGS)); print(remote_defaults.DEFAULT_SERVER_URL)'],
            cwd=REGISTRATION_PACKAGE_ROOT, env=env, text=True, capture_output=True)
        return result

    def test_public_defaults_ignore_machine_config_when_disabled(self):
        result = self._probe('')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(['{}', 'http://127.0.0.1:8188'], result.stdout.splitlines())

    def test_explicit_local_overrides_and_missing_config_failures(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps({'server_url': 'http://peer.example:8188', 'qwen_clip': 'folder/encoder.safetensors'}))
            result = self._probe(str(path))
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual('http://peer.example:8188', result.stdout.splitlines()[-1])
            path.unlink()
            self.assertNotEqual(0, self._probe(str(path)).returncode)

    def test_invalid_local_config_rejected_without_silent_fallback(self):
        import tempfile
        for payload in ['not json', '[]', '{"unknown": "value"}', '{"qwen_clip": 4}', '{"prompt_profile": []}']:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'settings.json'
                path.write_text(payload)
                self.assertNotEqual(0, self._probe(str(path)).returncode)

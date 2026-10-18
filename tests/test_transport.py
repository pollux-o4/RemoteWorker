from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import remoteworker.core.transport as http_remote_conditioning
from remoteworker.conditioning.workflow import extract_artifact_path
from remoteworker.core.protocol import validate_cleanup_capability


class http_FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit=None) -> bytes:
        if limit is not None:
            return self.body[:limit]
        return self.body


class HttpComfyApiTests(unittest.TestCase):
    def test_cancel_prefers_the_idempotent_jobs_endpoint_and_url_encodes_the_id(
        self,
    ) -> None:
        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)

        with patch.object(
            http_remote_conditioning,
            "urlopen",
            return_value=http_FakeResponse(b'{"cancelled":false}'),
        ) as open_request:
            api.cancel_prompt("prompt/id")

        request = open_request.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/api/jobs/prompt%2Fid/cancel"))
        self.assertEqual({}, json.loads(request.data))

    def test_cancel_falls_back_to_targeted_interrupt_and_queue_delete_on_404(
        self,
    ) -> None:
        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)
        missing = HTTPError(
            "http://127.0.0.1:8188/api/jobs/prompt-99/cancel",
            404,
            "missing",
            None,
            io.BytesIO(b"missing"),
        )

        with patch.object(
            http_remote_conditioning,
            "urlopen",
            side_effect=[missing, http_FakeResponse(b""), http_FakeResponse(b"")],
        ) as open_request:
            api.cancel_prompt("prompt-99")

        requests = [call.args[0] for call in open_request.call_args_list]
        self.assertTrue(requests[0].full_url.endswith("/api/jobs/prompt-99/cancel"))
        self.assertTrue(requests[1].full_url.endswith("/interrupt"))
        self.assertEqual({"prompt_id": "prompt-99"}, json.loads(requests[1].data))
        self.assertTrue(requests[2].full_url.endswith("/queue"))
        self.assertEqual({"delete": ["prompt-99"]}, json.loads(requests[2].data))

    def test_cancel_does_not_fallback_after_a_non_missing_jobs_endpoint_error(
        self,
    ) -> None:
        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)
        failed = HTTPError(
            "http://127.0.0.1:8188/api/jobs/prompt-99/cancel",
            500,
            "failed",
            None,
            io.BytesIO(b"failed"),
        )

        with patch.object(http_remote_conditioning, "urlopen", side_effect=failed) as open_request:
            with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                api.cancel_prompt("prompt-99")

        self.assertEqual(1, open_request.call_count)

    def test_history_wait_checks_the_caller_interrupt_signal_before_polling(
        self,
    ) -> None:
        class CallerInterrupted(BaseException):
            pass

        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)
        with patch.object(
            http_remote_conditioning,
            "_raise_if_caller_interrupted",
            side_effect=CallerInterrupted,
        ):
            with self.assertRaises(CallerInterrupted):
                api.wait_for_history("prompt-99")

    def test_upload_file_preserves_safe_media_filename_and_mime_type(self) -> None:
        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)
        response = http_FakeResponse(
            b'{"name":"audio-00.wav","subfolder":"remote-conditioning/request-42","type":"temp"}'
        )

        with patch.object(http_remote_conditioning, "urlopen", return_value=response) as open_request:
            path = api.upload_file(
                request_id="request-42",
                filename="audio-00.wav",
                data=b"RIFF",
                content_type="audio/wav",
            )

        request = open_request.call_args.args[0]
        self.assertEqual("remote-conditioning/request-42/audio-00.wav [temp]", path)
        self.assertIn(b'filename="audio-00.wav"', request.data)
        self.assertIn(b"Content-Type: audio/wav", request.data)

    def test_uses_prompt_history_and_view_with_a_registered_peer(self) -> None:
        api = http_remote_conditioning.HttpComfyApi(
            "https://nvi.example.internal:8188",
            timeout_seconds=30,
        )
        responses = [
            http_FakeResponse(b'{"prompt_id":"prompt-99"}'),
            http_FakeResponse(
                json.dumps(
                    {
                        "prompt-99": {
                            "status": {"status_str": "success"},
                            "outputs": {"save": {"text": ["remote-conditioning/request-42/cond_00001_.cond"]}},
                        }
                    }
                ).encode("utf-8")
            ),
            http_FakeResponse(b"trusted-cond"),
        ]

        with patch.object(http_remote_conditioning, "urlopen", side_effect=responses) as open_request:
            prompt_id = api.submit({"encode": {}})
            history = api.wait_for_history(prompt_id)
            artifact = extract_artifact_path(history, prompt_id, "request-42")
            body = api.download_artifact(artifact)

        self.assertEqual("prompt-99", prompt_id)
        self.assertEqual(b"trusted-cond", body)
        self.assertTrue(open_request.call_args_list[0].args[0].full_url.endswith("/prompt"))
        self.assertTrue(open_request.call_args_list[1].args[0].full_url.endswith("/history/prompt-99"))
        self.assertIn(
            "/view?filename=cond_00001_.cond",
            open_request.call_args_list[2].args[0].full_url,
        )

    def test_download_artifact_accepts_payload_above_64_mib(self) -> None:
        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)
        large_body = b"A" * (85 * 1024 * 1024)
        with patch.object(
            http_remote_conditioning,
            "urlopen",
            return_value=http_FakeResponse(large_body),
        ):
            body = api.download_artifact("remote-conditioning/request-42/cond_00001_.cond")
        self.assertEqual(len(large_body), len(body))

    def test_parse_env_positive_int(self) -> None:
        parse = http_remote_conditioning._parse_env_positive_int
        with patch.dict("os.environ", {"TEST_VAR": "1048576"}, clear=True):
            self.assertEqual(1048576, parse("TEST_VAR", 512))
        with patch.dict("os.environ", {"TEST_VAR": "not-an-int"}, clear=True):
            self.assertEqual(512, parse("TEST_VAR", 512))
        with patch.dict("os.environ", {"TEST_VAR": "-10"}, clear=True):
            self.assertEqual(512, parse("TEST_VAR", 512))
        with patch.dict("os.environ", {"TEST_VAR": "0"}, clear=True):
            self.assertEqual(512, parse("TEST_VAR", 512))
        with patch.dict("os.environ", {"TEST_VAR": "   "}, clear=True):
            self.assertEqual(512, parse("TEST_VAR", 512))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(512, parse("NON_EXISTENT_VAR", 512))

    def test_download_artifact_rejects_payload_exceeding_max_artifact_bytes(
        self,
    ) -> None:
        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)
        fake_limit = 512
        oversized = b"A" * (fake_limit + 10)
        with patch.object(http_remote_conditioning, "_MAX_ARTIFACT_BYTES", fake_limit):
            with patch.object(
                http_remote_conditioning,
                "urlopen",
                return_value=http_FakeResponse(oversized),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"Remote Conditioning artifact exceeds the 0 MiB limit \(0\.0 MiB received\)",
                ):
                    api.download_artifact("remote-conditioning/request-42/cond_00001_.cond")

    def test_upload_file_accepts_payload_above_64_mib(self) -> None:
        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)
        large_data = b"V" * (85 * 1024 * 1024)
        response = http_FakeResponse(
            b'{"name":"video-00.mp4","subfolder":"remote-conditioning/request-42","type":"temp"}'
        )
        with patch.object(http_remote_conditioning, "urlopen", return_value=response):
            path = api.upload_file(
                request_id="request-42",
                filename="video-00.mp4",
                data=large_data,
                content_type="video/mp4",
            )
        self.assertEqual("remote-conditioning/request-42/video-00.mp4 [temp]", path)

    def test_upload_file_rejects_payload_exceeding_max_artifact_bytes(self) -> None:
        api = http_remote_conditioning.HttpComfyApi("http://127.0.0.1:8188", timeout_seconds=30)
        fake_limit = 512
        oversized = b"V" * (fake_limit + 1)
        with patch.object(http_remote_conditioning, "_MAX_ARTIFACT_BYTES", fake_limit):
            with self.assertRaisesRegex(ValueError, "Remote Conditioning media upload arguments are invalid"):
                api.upload_file(
                    request_id="request-42",
                    filename="video-00.mp4",
                    data=oversized,
                    content_type="video/mp4",
                )


import remoteworker.core.transport as capability_remote_conditioning


class capability_FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit=None) -> bytes:
        return self.body


class ClipLoaderCapabilityTests(unittest.TestCase):
    def test_accepts_the_shared_vram_cleanup_contract(self) -> None:
        validate_cleanup_capability(
            {
                "input": {
                    "required": {
                        "offload_model": ["BOOLEAN", {"default": True}],
                        "offload_cache": ["BOOLEAN", {"default": True}],
                        "reset_execution_cache": ["BOOLEAN", {"default": False}],
                    },
                    "optional": {"anything": ["*", {}]},
                },
                "output": ["*"],
            }
        )

    def test_rejects_missing_vram_cleanup_contract(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "VRAMCleanup"):
            validate_cleanup_capability(
                {
                    "input": {"required": {}, "optional": {}},
                    "output": ["*"],
                }
            )

    def test_extracts_ordered_unique_clip_loader_options_from_object_info(self) -> None:
        object_info = {
            "CLIPLoader": {
                "input": {
                    "required": {
                        "clip_name": [
                            [
                                "first.safetensors",
                                "second.safetensors",
                                "first.safetensors",
                            ]
                        ],
                        "type": [["lumina2", "stable_diffusion", "lumina2"]],
                    },
                    "optional": {
                        "device": [["default", "cpu", "default"], {"advanced": True}],
                    },
                }
            }
        }

        self.assertEqual(
            {
                "clip_name": ["first.safetensors", "second.safetensors"],
                "clip_type": ["lumina2", "stable_diffusion"],
                "device": ["default", "cpu"],
            },
            capability_remote_conditioning.parse_clip_loader_options(object_info),
        )

    def test_rejects_missing_clip_loader_contract(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CLIPLoader object_info"):
            capability_remote_conditioning.parse_clip_loader_options({})

    def test_rejects_invalid_clip_loader_option_values(self) -> None:
        valid = {
            "CLIPLoader": {
                "input": {
                    "required": {
                        "clip_name": [["clip.safetensors"]],
                        "type": [["lumina2"]],
                    },
                    "optional": {"device": [["default"]]},
                }
            }
        }
        invalid_values = ([], "not-a-list", [""], ["   "], [7])

        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                object_info = {
                    "CLIPLoader": {
                        "input": {
                            "required": {
                                "clip_name": [invalid],
                                "type": valid["CLIPLoader"]["input"]["required"]["type"],
                            },
                            "optional": valid["CLIPLoader"]["input"]["optional"],
                        }
                    }
                }
                with self.assertRaisesRegex(RuntimeError, "clip_name"):
                    capability_remote_conditioning.parse_clip_loader_options(object_info)

    def test_http_api_fetches_and_parses_clip_loader_options(self) -> None:
        object_info = {
            "CLIPLoader": {
                "input": {
                    "required": {
                        "clip_name": [["remote-clip.safetensors"]],
                        "type": [["lumina2"]],
                    },
                    "optional": {"device": [["default", "cpu"]]},
                }
            }
        }
        api = capability_remote_conditioning.HttpComfyApi("https://nvi.example.internal:8188", timeout_seconds=30)

        with patch.object(
            capability_remote_conditioning,
            "urlopen",
            return_value=capability_FakeResponse(json.dumps(object_info).encode("utf-8")),
        ) as open_request:
            options = api.clip_loader_options()

        self.assertEqual(
            {
                "clip_name": ["remote-clip.safetensors"],
                "clip_type": ["lumina2"],
                "device": ["default", "cpu"],
            },
            options,
        )
        request = open_request.call_args.args[0]
        self.assertEqual(
            "https://nvi.example.internal:8188/object_info/CLIPLoader",
            request.full_url,
        )
        self.assertEqual("GET", request.method)


class RequestLifecycleTests(unittest.TestCase):
    def test_missing_prompt_id_fails_before_history_or_cancellation(self):
        from unittest.mock import Mock

        from remoteworker.core.execution import completed_artifacts
        api = Mock()
        api.submit.return_value = None
        with self.assertRaisesRegex(RuntimeError, "prompt_id"):
            with completed_artifacts(api, {}, "request-1", ()):
                self.fail("Invalid submission must not yield")
        api.wait_for_history.assert_not_called()
        api.cancel_prompt.assert_not_called()

    def test_decode_failure_cancels_only_submitted_prompt(self):
        from unittest.mock import Mock

        from remoteworker.core.execution import completed_artifacts
        api = Mock()
        api.submit.return_value = "prompt-1"
        api.wait_for_history.return_value = {"prompt-1": {
            "status": {"status_str": "success"},
            "outputs": {"save": {"text": ["remote-conditioning/request-1/result.cond"]}},
        }}
        with self.assertRaisesRegex(ValueError, "decode failed"):
            with completed_artifacts(api, {}, "request-1", (__import__("remoteworker.core.artifacts", fromlist=["CONDITIONING_ARTIFACT"]).CONDITIONING_ARTIFACT,)):
                raise ValueError("decode failed")
        api.cancel_prompt.assert_called_once_with("prompt-1")

    def test_partial_upload_failure_discards_request_without_masking_cause(self):
        from unittest.mock import Mock

        from remoteworker.core.execution import RequestInputs
        api = Mock()
        api.upload_image.side_effect = ["first.png [temp]", TimeoutError("second upload")]
        api.discard_inputs.side_effect = OSError("cleanup unavailable")
        with self.assertLogs("remoteworker.core.execution", level="WARNING") as caught:
            with self.assertRaisesRegex(TimeoutError, "second upload"):
                with RequestInputs(api, "request-1") as request:
                    request.upload_image(index=1, png=b"first")
                    request.upload_image(index=2, png=b"second")
        self.assertIn("discard", caught.output[0])
        api.discard_inputs.assert_called_once_with("request-1")

    def test_no_uploaded_inputs_do_not_trigger_cleanup(self):
        from unittest.mock import Mock

        from remoteworker.core.execution import RequestInputs
        api = Mock()
        with RequestInputs(api, "request-1"):
            pass
        api.discard_inputs.assert_not_called()

    def test_cleanup_and_cancel_keep_original_exception_when_warnings_are_errors(self):
        import warnings
        from unittest.mock import Mock

        from remoteworker.core.execution import RequestInputs, cancel_on_remote_failure
        api = Mock()
        api.cancel_prompt.side_effect = RuntimeError("cancel unavailable")
        api.discard_inputs.side_effect = RuntimeError("discard unavailable")
        with warnings.catch_warnings(), self.assertLogs("remoteworker.core.execution", level="WARNING") as caught:
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaisesRegex(TimeoutError, "original timeout"):
                with RequestInputs(api, "request-1") as request:
                    request.upload_file(filename="frame.png", data=b"png")
                    with cancel_on_remote_failure(api, "prompt-1"):
                        raise TimeoutError("original timeout")
        self.assertEqual(2, len(caught.output))


class PromptTransportTests(unittest.TestCase):
    def test_profile_lookup_validates_remote_records(self):
        from remoteworker.prompt.profiles import DEFAULT_PROFILE
        from remoteworker.prompt.transport import PromptComfyApi
        api = PromptComfyApi("http://127.0.0.1:8188", 30)
        with patch.object(api, "_json", return_value={"profiles": [DEFAULT_PROFILE.as_public_dict()]}):
            self.assertEqual(DEFAULT_PROFILE, api.get_prompt_enhancer_profile(DEFAULT_PROFILE.id))
        with patch.object(api, "_json", return_value={"profiles": [{"id": DEFAULT_PROFILE.id}]}):
            with self.assertRaises(ValueError):
                api.get_prompt_enhancer_profile(DEFAULT_PROFILE.id)

    def test_profile_management_preserves_endpoint_payload_contract(self):
        from remoteworker.prompt.profiles import DEFAULT_PROFILE
        from remoteworker.prompt.transport import PromptComfyApi
        api = PromptComfyApi("http://127.0.0.1:8188", 30)
        profile = DEFAULT_PROFILE.as_public_dict()
        with patch.object(api, "_json", return_value={"profile": profile}) as request:
            self.assertEqual(profile, api.upsert_prompt_enhancer_profile(profile))
            request.assert_called_once_with("/remote-prompt-enhancer/profiles", payload={"action": "upsert", "profile": profile})
        with patch.object(api, "_json", return_value={"deleted": True}) as request:
            self.assertTrue(api.delete_prompt_enhancer_profile(DEFAULT_PROFILE.id))
            request.assert_called_once_with("/remote-prompt-enhancer/profiles", payload={"action": "delete", "profile_id": DEFAULT_PROFILE.id})

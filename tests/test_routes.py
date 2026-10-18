from __future__ import annotations

import asyncio
import importlib
import json
import tempfile
import threading
import types
import unittest
from pathlib import Path


def conditioning_load_routes_module():
    return importlib.import_module("remoteworker.integration.discovery_routes")


class conditioning_FakeRequest:
    def __init__(self, payload=None, *, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    async def json(self):
        if self._error:
            raise self._error
        return self._payload


class conditioning_FakeResponse:
    def __init__(self, payload, *, status: int = 200) -> None:
        self.status = status
        self.payload = payload


class RemoteConditioningRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_registers_only_the_named_post_route_with_comfyui(self) -> None:
        routes = conditioning_load_routes_module()

        class FakeRoutes:
            def __init__(self) -> None:
                self.handlers = {}

            def post(self, path):
                def decorate(handler):
                    self.handlers[path] = handler
                    return handler

                return decorate

        fake_routes = FakeRoutes()
        fake_prompt_server = types.SimpleNamespace(
            instance=types.SimpleNamespace(routes=fake_routes),
        )

        self.assertTrue(routes.register_routes(fake_prompt_server))
        self.assertEqual(
            {
                "/remote-conditioning/clip-loader-options",
                "/remote-conditioning/vae-loader-options",
                "/remote-conditioning/discard-inputs",
            },
            set(fake_routes.handlers),
        )
        self.assertTrue(all(asyncio.iscoroutinefunction(handler) for handler in fake_routes.handlers.values()))

    async def test_discards_only_one_request_scoped_temp_directory(self) -> None:
        routes = conditioning_load_routes_module()
        with tempfile.TemporaryDirectory() as temp:
            request_dir = Path(temp) / "remote-conditioning" / "request-9"
            sibling = Path(temp) / "remote-conditioning" / "request-10"
            request_dir.mkdir(parents=True)
            sibling.mkdir()
            (request_dir / "image-0001.png").write_bytes(b"png")
            (sibling / "keep.png").write_bytes(b"keep")

            self.assertTrue(routes.discard_request_inputs("request-9", temp))
            self.assertFalse(request_dir.exists())
            self.assertTrue((sibling / "keep.png").exists())

            with self.assertRaises(ValueError):
                routes.discard_request_inputs("../escape", temp)

    async def test_returns_remote_clip_loader_options_from_same_origin_post(
        self,
    ) -> None:
        routes = conditioning_load_routes_module()

        class FakeApi:
            def __init__(self, server_url: str, timeout_seconds: int) -> None:
                self.connection = (server_url, timeout_seconds)

            def clip_loader_options(self):
                return {
                    "clip_name": ["qwen.safetensors"],
                    "clip_type": ["lumina2"],
                    "device": ["default", "cpu"],
                }

        handler = routes.create_clip_loader_options_handler(FakeApi, conditioning_FakeResponse)
        response = await handler(
            conditioning_FakeRequest(
                {
                    "server_url": "http://127.0.0.1:8188",
                    "timeout_seconds": 5,
                }
            )
        )

        self.assertEqual(200, response.status)
        self.assertEqual(
            {
                "clip_name": ["qwen.safetensors"],
                "clip_type": ["lumina2"],
                "device": ["default", "cpu"],
            },
            response.payload,
        )

    async def test_returns_remote_vae_loader_options_from_same_origin_post(
        self,
    ) -> None:
        routes = conditioning_load_routes_module()

        class FakeApi:
            def __init__(self, server_url: str, timeout_seconds: int) -> None:
                self.connection = (server_url, timeout_seconds)

            def vae_loader_options(self):
                return {
                    "vae_name": [
                        "remote-models\\minimax_h3_video_vae_fp16.safetensors",
                        "minimax_h3_audio_vae_fp32.safetensors",
                    ],
                }

        handler = routes.create_vae_loader_options_handler(FakeApi, conditioning_FakeResponse)
        response = await handler(
            conditioning_FakeRequest(
                {
                    "server_url": "http://127.0.0.1:8188",
                    "timeout_seconds": 5,
                }
            )
        )

        self.assertEqual(200, response.status)
        self.assertEqual(
            {
                "vae_name": [
                    "remote-models\\minimax_h3_video_vae_fp16.safetensors",
                    "minimax_h3_audio_vae_fp32.safetensors",
                ],
            },
            response.payload,
        )

    async def test_rejects_invalid_request_before_contacting_the_peer(self) -> None:
        routes = conditioning_load_routes_module()

        class UnexpectedApi:
            def __init__(self, *_args, **_kwargs) -> None:
                raise AssertionError("invalid input must not create an upstream client")

        handler = routes.create_clip_loader_options_handler(UnexpectedApi, conditioning_FakeResponse)
        invalid_requests = [
            conditioning_FakeRequest(error=json.JSONDecodeError("bad JSON", "{", 1)),
            conditioning_FakeRequest([]),
            conditioning_FakeRequest({"timeout_seconds": 5}),
            conditioning_FakeRequest({"server_url": "http://127.0.0.1:8188", "timeout_seconds": True}),
            conditioning_FakeRequest({"server_url": "http://127.0.0.1:8188", "timeout_seconds": 4}),
            conditioning_FakeRequest({"server_url": "http://127.0.0.1:8188", "timeout_seconds": 31}),
            conditioning_FakeRequest(
                {
                    "server_url": "http://127.0.0.1:8188",
                    "timeout_seconds": 5,
                    "path": "/prompt",
                }
            ),
        ]

        for request in invalid_requests:
            with self.subTest(request=request):
                response = await handler(request)
                self.assertEqual(400, response.status)
                self.assertEqual("invalid_request", response.payload["error"]["code"])
                self.assertTrue(response.payload["error"]["message"])

    async def test_rejects_a_non_origin_server_url_as_invalid_input(self) -> None:
        routes = conditioning_load_routes_module()
        handler = routes.create_clip_loader_options_handler(
            routes.HttpComfyApi,
            conditioning_FakeResponse,
        )

        response = await handler(
            conditioning_FakeRequest(
                {
                    "server_url": "http://127.0.0.1:8188/prompt",
                    "timeout_seconds": 5,
                }
            )
        )

        self.assertEqual(400, response.status)
        self.assertEqual("invalid_request", response.payload["error"]["code"])
        self.assertIn("origin", response.payload["error"]["message"])

    async def test_maps_peer_failures_to_a_clear_bad_gateway_response(self) -> None:
        routes = conditioning_load_routes_module()

        class FailingApi:
            def __init__(self, _server_url: str, _timeout_seconds: int) -> None:
                pass

            def clip_loader_options(self):
                raise RuntimeError("Remote Conditioning returned invalid JSON from /object_info/CLIPLoader")

        handler = routes.create_clip_loader_options_handler(FailingApi, conditioning_FakeResponse)
        response = await handler(
            conditioning_FakeRequest(
                {
                    "server_url": "http://127.0.0.1:8188",
                    "timeout_seconds": 5,
                }
            )
        )

        self.assertEqual(502, response.status)
        self.assertEqual("upstream_failure", response.payload["error"]["code"])
        self.assertIn("invalid JSON", response.payload["error"]["message"])

    async def test_rejects_a_malformed_peer_capability_response(self) -> None:
        routes = conditioning_load_routes_module()

        class MalformedApi:
            def __init__(self, _server_url: str, _timeout_seconds: int) -> None:
                pass

            def clip_loader_options(self):
                return {"clip_name": ["qwen.safetensors"]}

        handler = routes.create_clip_loader_options_handler(MalformedApi, conditioning_FakeResponse)
        response = await handler(
            conditioning_FakeRequest(
                {
                    "server_url": "http://127.0.0.1:8188",
                    "timeout_seconds": 5,
                }
            )
        )

        self.assertEqual(502, response.status)
        self.assertEqual("upstream_failure", response.payload["error"]["code"])
        self.assertIn("clip_type", response.payload["error"]["message"])

    async def test_rejects_a_malformed_peer_vae_capability_response(self) -> None:
        routes = conditioning_load_routes_module()

        class MalformedApi:
            def __init__(self, _server_url: str, _timeout_seconds: int) -> None:
                pass

            def vae_loader_options(self):
                return {"vae_name": []}

        handler = routes.create_vae_loader_options_handler(MalformedApi, conditioning_FakeResponse)
        response = await handler(
            conditioning_FakeRequest(
                {
                    "server_url": "http://127.0.0.1:8188",
                    "timeout_seconds": 5,
                }
            )
        )

        self.assertEqual(502, response.status)
        self.assertEqual("upstream_failure", response.payload["error"]["code"])
        self.assertIn("vae_name", response.payload["error"]["message"])

    async def test_peer_io_does_not_block_the_comfyui_event_loop(self) -> None:
        routes = conditioning_load_routes_module()
        started = threading.Event()
        release = threading.Event()
        observed_while_blocked = False

        class BlockingApi:
            def __init__(self, _server_url: str, _timeout_seconds: int) -> None:
                pass

            def clip_loader_options(self):
                started.set()
                release.wait(1)
                return {"clip_name": ["a"], "clip_type": ["b"], "device": ["default"]}

        async def probe_event_loop() -> None:
            nonlocal observed_while_blocked
            while not started.is_set():
                await asyncio.sleep(0.001)
            observed_while_blocked = not release.is_set()
            release.set()

        handler = routes.create_clip_loader_options_handler(BlockingApi, conditioning_FakeResponse)
        request = conditioning_FakeRequest({"server_url": "http://127.0.0.1:8188", "timeout_seconds": 5})
        response, _ = await asyncio.gather(handler(request), probe_event_loop())

        self.assertTrue(observed_while_blocked)
        self.assertEqual(200, response.status)


def profile_load_routes():
    return importlib.import_module("remoteworker.integration.profile_routes")


class profile_FakeRequest:
    def __init__(self, method: str, payload=None):
        self.method = method
        self.payload = payload

    async def json(self):
        return self.payload


class profile_FakeResponse:
    def __init__(self, payload, *, status=200):
        self.payload = payload
        self.status = status


class PromptEnhancerRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_nvi_route_lists_describes_upserts_and_deletes_profiles(self) -> None:
        routes = profile_load_routes()
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "profiles.json"
            handler = routes.create_profiles_handler(
                lambda: routes.PromptEnhancerProfileStore(registry),
                profile_FakeResponse,
            )
            listed = await handler(profile_FakeRequest("GET"))
            self.assertEqual(200, listed.status)
            self.assertEqual("example", listed.payload["profiles"][0]["id"])
            self.assertTrue(listed.payload["profiles"][0]["description"])

            created = await handler(
                profile_FakeRequest(
                    "POST",
                    {
                        "action": "upsert",
                        "profile": {
                            "id": "video-clean",
                            "description": "Keep motion instructions explicit.",
                            "clip_name": "qwen3vl_4b_fp8_scaled(krea).safetensors",
                            "clip_type": "krea2",
                            "device": "default",
                            "max_length": 256,
                            "system_prompt": "Return only one enhanced prompt.",
                            "accepted_media": ["image"],
                        },
                    },
                )
            )
            self.assertEqual(200, created.status)
            self.assertEqual("video-clean", created.payload["profile"]["id"])
            removed = await handler(profile_FakeRequest("POST", {"action": "delete", "profile_id": "video-clean"}))
            self.assertEqual({"deleted": True}, removed.payload)

    async def test_same_origin_proxy_returns_only_valid_described_profiles(
        self,
    ) -> None:
        routes = profile_load_routes()

        class FakeApi:
            def __init__(self, _url, _timeout):
                pass

            def list_prompt_enhancer_profiles(self):
                return [
                    {
                        "id": "example",
                        "description": "Fast profile.",
                        "clip_name": "qwen.safetensors",
                        "clip_type": "krea2",
                        "device": "default",
                        "max_length": 256,
                    }
                ]

        handler = routes.create_profile_options_proxy_handler(FakeApi, profile_FakeResponse)
        response = await handler(
            profile_FakeRequest("POST", {"server_url": "http://127.0.0.1:8188", "timeout_seconds": 5})
        )
        self.assertEqual(200, response.status)
        self.assertEqual("Fast profile.", response.payload["profiles"][0]["description"])

    async def test_same_origin_manager_forwards_only_a_valid_upsert(self) -> None:
        routes = profile_load_routes()

        class FakeApi:
            def __init__(self, _url, _timeout):
                pass

            def upsert_prompt_enhancer_profile(self, profile):
                return profile

        handler = routes.create_profile_manage_proxy_handler(FakeApi, profile_FakeResponse)
        response = await handler(
            profile_FakeRequest(
                "POST",
                {
                    "server_url": "http://127.0.0.1:8188",
                    "timeout_seconds": 5,
                    "action": "upsert",
                    "profile": {
                        "id": "fast",
                        "description": "Fast profile.",
                        "clip_name": "qwen.safetensors",
                        "clip_type": "krea2",
                        "device": "default",
                        "max_length": 256,
                        "system_prompt": "Return only one enhanced prompt.",
                        "accepted_media": ["image"],
                    },
                },
            )
        )
        self.assertEqual(200, response.status)
        self.assertEqual("fast", response.payload["profile"]["id"])

    async def test_registers_nvi_crud_and_amd_same_origin_proxy_routes(self) -> None:
        routes = profile_load_routes()

        class FakeRoutes:
            def __init__(self):
                self.handlers = {}

            def get(self, path):
                return lambda handler: self.handlers.setdefault(("GET", path), handler) or handler

            def post(self, path):
                return lambda handler: self.handlers.setdefault(("POST", path), handler) or handler

        fake = FakeRoutes()
        server = types.SimpleNamespace(instance=types.SimpleNamespace(routes=fake))
        self.assertTrue(routes.register_routes(server))
        self.assertEqual(
            {
                ("GET", "/remote-prompt-enhancer/profiles"),
                ("POST", "/remote-prompt-enhancer/profiles"),
                ("POST", "/remote-prompt-enhancer/profile-options"),
                ("POST", "/remote-prompt-enhancer/profiles/manage"),
            },
            set(fake.handlers),
        )

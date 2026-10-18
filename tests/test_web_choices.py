import json
import shutil
import textwrap
import unittest

from web_harness import REMOTE_MINIMAX_H3, REMOTE_MINIMAX_H3_IMAGE_TO_VIDEO, run_web


class WebChoicesTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_remote_options_preserve_valid_values_and_fallback_atomically(self) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { reconcileRemoteOptions } from "./custom/web/remote_choices.js";

                                const current = { clip_name: "keep.safetensors", clip_type: "old", device: "cuda" };
                                const remote = {
                                  clip_name: ["first.safetensors", "keep.safetensors"],
                                  clip_type: ["lumina2", "flux"],
                                  device: ["default", "cpu"],
                                };
                                process.stdout.write(JSON.stringify(reconcileRemoteOptions(current, remote)));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        self.assertEqual(
            {
                "clip_name": "keep.safetensors",
                "clip_type": "lumina2",
                "device": "default",
            },
            json.loads(completed.stdout),
        )

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_remote_options_turn_string_widgets_into_selectors_after_refresh(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { applyRemoteOptions } from "./custom/web/remote_choices.js";

                                const widgets = [
                                  { name: "remote_text_encoder", type: "text", value: "hot-tier\\\\future.safetensors", options: {} },
                                  { name: "clip_type", type: "text", value: "krea2", options: {} },
                                  { name: "device", type: "text", value: "default", options: {} },
                                ];
                                applyRemoteOptions({ widgets }, {
                                  clip_name: ["hot-tier\\\\future.safetensors", "other.safetensors"],
                                  clip_type: ["krea2"],
                                  device: ["default"],
                                });
                                process.stdout.write(JSON.stringify(widgets));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        widgets = json.loads(completed.stdout)
        self.assertEqual(["combo", "combo", "combo"], [widget["type"] for widget in widgets])
        self.assertEqual("hot-tier\\future.safetensors", widgets[0]["value"])
        self.assertEqual(
            ["hot-tier\\future.safetensors", "other.safetensors"],
            widgets[0]["options"]["values"],
        )

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_remote_options_support_minimax_clip_name_widgets_from_node_schemas(
        self,
    ) -> None:
        node_types = {
            "RemoteMiniMaxH3ReferenceEncode": REMOTE_MINIMAX_H3.RemoteMiniMaxH3ReferenceEncode,
            "RemoteMiniMaxH3ImageToVideo": REMOTE_MINIMAX_H3_IMAGE_TO_VIDEO.RemoteMiniMaxH3ImageToVideo,
        }
        node_contracts = []
        for node_name, node_type in node_types.items():
            required = node_type.INPUT_TYPES()["required"]
            widgets = [
                {
                    "name": name,
                    "type": "combo",
                    "value": required[name][0][0] or "example-model.safetensors",
                    "options": {},
                }
                for name in ("clip_name", "clip_type", "device")
            ]
            node_contracts.append(
                {
                    "node_name": node_name,
                    "widgets": widgets,
                    "remote": {
                        "clip_name": [widgets[0]["value"], "other_minimax.safetensors"],
                        "clip_type": [widgets[1]["value"]],
                        "device": [widgets[2]["value"]],
                    },
                }
            )

        completed = run_web(
            textwrap.dedent(
                """
                                import { applyRemoteOptions } from "./custom/web/remote_choices.js";

                                const contracts = JSON.parse(process.argv[2]);
                                const results = contracts.map(({ node_name, widgets, remote }) => {
                                  const count = applyRemoteOptions({ widgets }, remote);
                                  const afterSuccess = structuredClone(widgets);
                                  try {
                                    applyRemoteOptions({ widgets }, { ...remote, clip_type: [] });
                                  } catch (_error) {}
                                  return { node_name, count, widgets, afterSuccess };
                                });
                                process.stdout.write(JSON.stringify(results));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
            args=[json.dumps(node_contracts)],
        )

        results = json.loads(completed.stdout)
        self.assertEqual(set(node_types), {result["node_name"] for result in results})
        for result in results:
            self.assertEqual(2, result["count"])
            self.assertEqual(
                ["combo", "combo", "combo"],
                [widget["type"] for widget in result["widgets"]],
            )
            self.assertEqual(result["afterSuccess"], result["widgets"])
            self.assertEqual(
                "other_minimax.safetensors",
                result["widgets"][0]["options"]["values"][1],
            )

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_invalid_remote_response_does_not_partially_mutate_widgets(self) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { applyRemoteOptions } from "./custom/web/remote_choices.js";

                                const widgets = [
                                  { name: "remote_text_encoder", type: "combo", value: "keep.safetensors", options: {} },
                                  { name: "clip_type", type: "combo", value: "lumina2", options: {} },
                                  { name: "device", type: "combo", value: "default", options: {} },
                                ];
                                try {
                                  applyRemoteOptions({ widgets }, {
                                    clip_name: ["replacement.safetensors"], clip_type: [], device: ["cpu"],
                                  });
                                } catch (_error) {}
                                process.stdout.write(JSON.stringify(widgets));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        widgets = json.loads(completed.stdout)
        self.assertEqual(["combo", "combo", "combo"], [widget["type"] for widget in widgets])
        self.assertEqual(
            ["keep.safetensors", "lumina2", "default"],
            [widget["value"] for widget in widgets],
        )

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_remote_vae_options_replace_saved_value_when_missing_from_peer_list(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { applyRemoteVaeOptions } from "./custom/web/remote_choices.js";

                                const widgets = [
                                  { name: "vae_name", value: "remote-models\\\\old-video.vae", options: {}, callback() {} },
                                  { name: "audio_vae_name", value: "remote-models\\\\old-audio.vae", options: {}, callback() {} },
                                ];
                                const count = applyRemoteVaeOptions({ widgets }, {
                                  vae_name: ["remote-models\\\\new-video.vae", "remote-models\\\\new-audio.vae"],
                                });
                                process.stdout.write(JSON.stringify({ count, widgets }));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        result = json.loads(completed.stdout)
        self.assertEqual(2, result["count"])
        self.assertEqual(
            ["remote-models\\new-video.vae", "remote-models\\new-video.vae"],
            [widget["value"] for widget in result["widgets"]],
        )
        self.assertEqual(
            ["remote-models\\new-video.vae", "remote-models\\new-audio.vae"],
            result["widgets"][0]["options"]["values"],
        )

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_invalid_remote_vae_response_does_not_mutate_widgets(self) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { applyRemoteVaeOptions } from "./custom/web/remote_choices.js";

                                const widgets = [{ name: "vae_name", value: "remote-models\\\\keep.vae", options: { values: ["remote-models\\\\keep.vae"] } }];
                                let failed = false;
                                try {
                                  applyRemoteVaeOptions({ widgets }, { vae_name: [] });
                                } catch (_error) { failed = true; }
                                process.stdout.write(JSON.stringify({ failed, widgets }));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        result = json.loads(completed.stdout)
        self.assertTrue(result["failed"])
        self.assertEqual("remote-models\\keep.vae", result["widgets"][0]["value"])
        self.assertEqual(["remote-models\\keep.vae"], result["widgets"][0]["options"]["values"])

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_vue_node_definition_receives_remote_profile_combo_before_render(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { applyPromptProfilesToVueDefs } from "./custom/web/remote_profiles.js";
                                const definitions = [{ name: "RemotePromptEnhancer", input: {
                                  required: { profile_id: [["stale"], {}], server_url: ["STRING", { default: "http://127.0.0.1:8188" }] }
                                } }];
                                applyPromptProfilesToVueDefs(definitions, { profiles: [
                                  { id: "example-small", description: "INT4", clip_name: "a", clip_type: "krea2", device: "default", max_length: 256, system_prompt: "x", accepted_media: ["image"] },
                                  { id: "example", description: "INT8", clip_name: "b", clip_type: "krea2", device: "default", max_length: 256, system_prompt: "x", accepted_media: ["image"] },
                                ] });
                                process.stdout.write(JSON.stringify(definitions[0].input.required.profile_id));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        options, metadata = json.loads(completed.stdout)
        self.assertEqual(["example-small", "example", "Manage profiles..."], options)
        self.assertIn("example-small: INT4", metadata["tooltip"])

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_definition_loading_hook_populates_profiles_before_vue_registration(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import "./custom/web/remote_conditioning.js";
                                import { extension } from "./scripts/app.js";
                                const definitions = { RemotePromptEnhancer: {
                                  name: "RemotePromptEnhancer", input: { required: { profile_id: [["stale"], {}], server_url: ["STRING", { default: "http://127.0.0.1:8188" }] } }
                                } };
                                await extension.addCustomNodeDefs(definitions);
                                process.stdout.write(JSON.stringify(definitions.RemotePromptEnhancer.input.required.profile_id[0]));
                                """
            ),
            app="export let extension; export const app = { registerExtension(value) { extension = value; }, graph: { setDirtyCanvas() {} } };",
            api="export const api = { fetchApi: async () => ({ ok: true, json: async () => ({ profiles: ["
            "{ id: 'example-small', description: 'INT4', clip_name: 'a', clip_type: 'krea2', device: 'default', max_length: 256, system_prompt: 'x', accepted_media: ['image'] },"
            "{ id: 'example', description: 'INT8', clip_name: 'b', clip_type: 'krea2', device: 'default', max_length: 256, system_prompt: 'x', accepted_media: ['image'] }"
            "] }) }) };",
        )

        self.assertEqual(
            ["example-small", "example", "Manage profiles..."],
            json.loads(completed.stdout),
        )

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_mixed_invalid_remote_values_fail_fast_without_mutating_any_widget(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { applyRemoteOptions } from "./custom/web/remote_choices.js";

                                const widgets = [
                                  { name: "remote_text_encoder", type: "combo", value: "keep.safetensors", options: {} },
                                  { name: "clip_type", type: "combo", value: "lumina2", options: {} },
                                  { name: "device", type: "combo", value: "default", options: {} },
                                ];
                                let failed = false;
                                try {
                                  applyRemoteOptions({ widgets }, {
                                    clip_name: ["replacement.safetensors"],
                                    clip_type: ["flux", null],
                                    device: ["cpu"],
                                  });
                                } catch (_error) { failed = true; }
                                process.stdout.write(JSON.stringify({ failed, widgets }));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        result = json.loads(completed.stdout)
        self.assertTrue(result["failed"])
        self.assertEqual(
            ["combo", "combo", "combo"],
            [widget["type"] for widget in result["widgets"]],
        )
        self.assertEqual(
            ["keep.safetensors", "lumina2", "default"],
            [widget["value"] for widget in result["widgets"]],
        )

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_prompt_enhancer_keeps_autogrown_images_before_video_inputs(self) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { syncPromptMediaInputs } from "./custom/web/remote_sockets.js";
                                const node = {
                                  widgets: [{ name: "profile_id", value: "minimax" }],
                                  _remotePromptProfiles: [{ id: "minimax", accepted_media: ["image", "video"] }],
                                  inputs: [], size: [100, 100],
                                  addInput(name, type) { this.inputs.push({ name, type, link: null }); },
                                  removeInput(index) { this.inputs.splice(index, 1); },
                                  computeSize() { return this.size; }, setSize() {},
                                };
                                syncPromptMediaInputs(node);
                                node.inputs.find((input) => input.name === "image_0").link = 10;
                                syncPromptMediaInputs(node);
                                process.stdout.write(JSON.stringify(node.inputs.map((input) => input.name)));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        self.assertEqual(["image_0", "image_1", "video_0"], json.loads(completed.stdout))

    def test_changed_server_during_body_read_preserves_choices_for_all_refreshes(self):
        run_web('''
            import assert from "node:assert/strict";
            import { api } from "./scripts/api.js";
            import { refreshRemoteOptions } from "./custom/web/remote_choices.js";
            import { refreshPromptProfiles } from "./custom/web/remote_profiles.js";
            import { refreshPeerChoices } from "./custom/web/remote_qwen_image21.js";
            for (const refresh of [refreshRemoteOptions, refreshPromptProfiles,
                (node, button) => refreshPeerChoices(node, button, "encoder")]) {
                const server = { name: "server_url", value: "http://first:8188" };
                const widgets = [server, { name: "timeout_seconds", value: 100 },
                    { name: "remote_text_encoder", value: "kept", options: { values: ["kept"] } }];
                const button = {};
                api.fetchApi = async (path, options) => {
                    assert.equal(JSON.parse(options.body).timeout_seconds, 30);
                    return { ok: true, async json() {
                        server.value = "http://second:8188";
                        return { clip_name: ["stale"], clip_type: ["type"], device: ["cpu"] };
                    } };
                };
                await refresh({ widgets }, button);
                assert.equal(widgets[2].value, "kept");
                assert.deepEqual(widgets[2].options.values, ["kept"]);
                assert.match(button.name, /변경됨/);
            }
        ''')

    def test_qwen_rejects_empty_choices_and_releases_busy_state(self):
        run_web('''
            import assert from "node:assert/strict";
            import { api } from "./scripts/api.js";
            import { refreshPeerChoices } from "./custom/web/remote_qwen_image21.js";
            const target = { name: "remote_text_encoder", value: "kept", options: { values: ["kept"] } };
            const node = { widgets: [target, { name: "server_url", value: "http://peer:8188" },
                { name: "timeout_seconds", value: 120 }] };
            api.fetchApi = async () => ({ ok: true, json: async () => ({ clip_name: [] }) });
            const button = {};
            await refreshPeerChoices(node, button, "encoder");
            assert.equal(target.value, "kept");
            assert.deepEqual(target.options.values, ["kept"]);
            assert.equal(button._remoteQwenRefreshInFlight, false);
            assert.match(button.name, /조회 실패/);
        ''')

    def test_profile_description_fits_node_and_keeps_full_tooltip(self):
        run_web('''
            import assert from "node:assert/strict";
            import { updateProfileDescription } from "./custom/web/remote_profiles.js";
            const description = "Reference profile description ".repeat(8);
            const button = { options: { serialize: false } };
            const node = { size: [220, 300], widgets: [{ name: "profile_id", value: "test" }],
                _remotePromptProfiles: [{ id: "test", description }], _remotePromptDescriptionButton: button };
            updateProfileDescription(node);
            assert.ok(button.name.length <= 24);
            assert.equal(button.options.tooltip, description);
            assert.equal(button.options.serialize, false);
        ''')

    def test_switching_profiles_preserves_each_undiscovered_model_without_duplicates(self):
        run_web('''
            import assert from "node:assert/strict";
            import { makeSelect, selectProfileValue } from "./custom/web/remote_profile_dialog.js";
            globalThis.document = { createElement(tag) {
                return tag === "select" ? { options: [], append(option) { this.options.push(option); },
                    set value(value) { this.selected = this.options.some(option => option.value === value) ? value : ""; },
                    get value() { return this.selected; } } : {};
            } };
            const select = makeSelect(["available"], "missingA");
            assert.equal(select.value, "missingA");
            selectProfileValue(select, "missingB");
            assert.equal(select.value, "missingB");
            assert.match(select.options.find(option => option.value === "missingB").textContent, /서버 목록에 없음/);
            selectProfileValue(select, "missingA");
            assert.equal(select.value, "missingA");
            assert.equal(select.options.length, 3);
        ''')

    def test_korean_description_uses_canvas_measurements_and_restores_font(self):
        run_web('''
            import assert from "node:assert/strict";
            import { app } from "./scripts/app.js";
            import { updateProfileDescription } from "./custom/web/remote_profiles.js";
            const width = text => [...text].reduce((size, char) => size + (char.codePointAt(0) > 255 ? 14 : 8), 0);
            app.canvas = { inner_text_font: "14px Arial", ctx: { font: "original", measureText(text) {
                assert.equal(this.font, "14px Arial"); return { width: width(text) };
            } } };
            const description = "긴 한글 프로필 설명을 노드 밖으로 넘치지 않게 표시합니다. ".repeat(5);
            const button = {};
            updateProfileDescription({ size: [220, 300], widgets: [{ name: "profile_id", value: "test" }],
                _remotePromptProfiles: [{ id: "test", description }], _remotePromptDescriptionButton: button });
            assert.ok(width(button.name) <= 180);
            assert.equal(button.options.tooltip, description);
            assert.equal(app.canvas.ctx.font, "original");
        ''')

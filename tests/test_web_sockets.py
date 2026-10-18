import json
import shutil
import textwrap
import unittest

from web_harness import run_web


class WebSocketsTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_startup_profile_metadata_creates_the_first_media_socket_without_manual_refresh(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import "./custom/web/remote_conditioning.js";
                                import { extension } from "./scripts/app.js";
                                await extension.addCustomNodeDefs({ RemotePromptEnhancer: {
                                  name: "RemotePromptEnhancer", input: { required: { profile_id: [["stale"], {}], server_url: ["STRING", { default: "http://127.0.0.1:8188" }] } }
                                } });
                                function Node() {
                                  this.widgets = [
                                    { name: "profile_id", value: "image-only", options: {}, callback() {} },
                                    { name: "server_url", value: "http://127.0.0.1:8188" },
                                    { name: "timeout_seconds", value: 180 },
                                    { name: "profile_manager", value: "" },
                                  ];
                                  this.inputs = [{ name: "profile_manager", type: "*", link: null }]; this.size = [100, 100];
                                }
                                Node.prototype.addWidget = function(type, name, value, callback) {
                                  const widget = { type, name, value, callback, options: {} }; this.widgets.push(widget); return widget;
                                };
                                Node.prototype.addInput = function(name, type) { this.inputs.push({ name, type, link: null }); };
                                Node.prototype.removeInput = function(index) { this.inputs.splice(index, 1); };
                                Node.prototype.removeWidget = function(index) { this.widgets.splice(index, 1); };
                                Node.prototype.computeSize = function() { return this.size; };
                                Node.prototype.setSize = function() {};
                                await extension.beforeRegisterNodeDef(Node, { name: "RemotePromptEnhancer" });
                                const node = new Node();
                                node.onNodeCreated();
                                await new Promise((resolve) => queueMicrotask(resolve));
                                process.stdout.write(JSON.stringify({
                                  inputs: node.inputs.map((input) => input.name),
                                  hasLegacyWidget: node.widgets.some((widget) => widget.name === "profile_manager"),
                                }));
                                """
            ),
            app="export let extension; export const app = { registerExtension(value) { extension = value; }, graph: { setDirtyCanvas() {} } };",
            api="export const api = { fetchApi: async () => ({ ok: true, json: async () => ({ profiles: ["
            "{ id: 'image-only', description: 'Image profile', clip_name: 'a', clip_type: 'krea2', device: 'default', max_length: 256, system_prompt: 'x', accepted_media: ['image'] }"
            "] }) }) };",
        )

        result = json.loads(completed.stdout)
        self.assertEqual(["image_0"], result["inputs"])
        self.assertFalse(result["hasLegacyWidget"])

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_saved_minimax_vae_override_inputs_become_widget_backed_override_inputs(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { normalizeMiniMaxVaeInputs } from "./custom/web/remote_sockets.js";
                                import { app } from "./scripts/app.js";

                                const node = { inputs: [
                                  { name: "vae_name_override", type: "STRING", link: null },
                                  { name: "audio_vae_name_override", type: "STRING", link: null },
                                  { name: "ref_image_0", type: "IMAGE", link: null },
                                ], widgets: [
                                  { name: "vae_name", type: "combo" },
                                  { name: "audio_vae_name", type: "combo" },
                                ], addInput(name, type, options) {
                                  this.inputs.push({ name, type, widget: options.widget, link: null });
                                }, removeInput(index) {
                                  this.inputs.splice(index, 1);
                                } };
                                const changed = normalizeMiniMaxVaeInputs(node);
                                process.stdout.write(JSON.stringify({ changed, dirty: app.graph.dirty, inputs: node.inputs }));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { dirty: false, setDirtyCanvas() { this.dirty = true; } } };",
            api="export const api = {};",
        )

        result = json.loads(completed.stdout)
        self.assertTrue(result["changed"])
        self.assertTrue(result["dirty"])
        self.assertEqual(
            ["vae_name_override", "audio_vae_name_override", "ref_image_0"],
            [item["name"] for item in result["inputs"]],
        )
        self.assertEqual(["*", "*", "IMAGE"], [item["type"] for item in result["inputs"]])
        self.assertEqual(
            ["vae_name", "audio_vae_name"],
            [item["widget"]["name"] for item in result["inputs"][:2]],
        )
        self.assertEqual(
            ["vae_name", "audio_vae_name"],
            [item["label"] for item in result["inputs"][:2]],
        )

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_saved_minimax_primary_vae_link_is_moved_to_override_input(self) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { normalizeMiniMaxVaeInputs } from "./custom/web/remote_sockets.js";

                                const node = { graph: { links: { 99: { target_slot: 0 } } }, inputs: [
                                  { name: "vae_name", type: "*", link: 99 },
                                ], widgets: [
                                  { name: "vae_name", type: "combo" },
                                ], addInput(name, type, options) {
                                  this.inputs.push({ name, type, widget: options.widget, link: null });
                                }, removeInput(index) {
                                  this.inputs.splice(index, 1);
                                } };
                                normalizeMiniMaxVaeInputs(node);
                                process.stdout.write(JSON.stringify({ inputs: node.inputs, link: node.graph.links[99] }));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        result = json.loads(completed.stdout)
        self.assertEqual(["vae_name_override"], [item["name"] for item in result["inputs"]])
        self.assertEqual(["*"], [item["type"] for item in result["inputs"]])
        self.assertEqual([99], [item["link"] for item in result["inputs"]])
        self.assertEqual(["vae_name"], [item["widget"]["name"] for item in result["inputs"]])
        self.assertEqual(0, result["link"]["target_slot"])

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_prompt_enhancer_media_sockets_follow_the_selected_profile_without_count_widgets(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { syncPromptMediaInputs } from "./custom/web/remote_sockets.js";
                                const node = {
                                  widgets: [{ name: "profile_id", value: "image-only" }],
                                  _remotePromptProfiles: [
                                    { id: "image-only", accepted_media: ["image"] },
                                    { id: "minimax", accepted_media: ["image", "video"] },
                                  ],
                                  inputs: [], size: [100, 100],
                                  addInput(name, type) { this.inputs.push({ name, type, link: null }); },
                                  removeInput(index) { this.inputs.splice(index, 1); },
                                  computeSize() { return this.size; }, setSize() {},
                                };
                                syncPromptMediaInputs(node);
                                node.inputs.find((input) => input.name === "image_0").link = 10;
                                syncPromptMediaInputs(node);
                                node.widgets[0].value = "minimax";
                                syncPromptMediaInputs(node);
                                node.inputs.find((input) => input.name === "video_0").link = 11;
                                syncPromptMediaInputs(node);
                                process.stdout.write(JSON.stringify(node.inputs.map((input) => input.name)));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        self.assertEqual(["image_0", "image_1", "video_0", "video_1"], json.loads(completed.stdout))

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_profile_refresh_materializes_media_socket_on_an_existing_node(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { applyPromptProfiles } from "./custom/web/remote_profiles.js";
                                const node = {
                                  widgets: [{ name: "profile_id", value: "image-only", options: {} }],
                                  inputs: [], size: [100, 100],
                                  addInput(name, type) { this.inputs.push({ name, type, link: null }); },
                                  removeInput(index) { this.inputs.splice(index, 1); },
                                  computeSize() { return this.size; }, setSize() {},
                                };
                                applyPromptProfiles(node, { profiles: [{
                                  id: "image-only", description: "Image profile", accepted_media: ["image"]
                                }] });
                                process.stdout.write(JSON.stringify(node.inputs.map((input) => input.name)));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        self.assertEqual(["image_0"], json.loads(completed.stdout))

    @unittest.skipUnless(
        shutil.which("node"),
        "Node.js is needed for the browser-independent JS contract test",
    )
    def test_minimax_sockets_autogrow_after_a_connection_without_count_widgets(
        self,
    ) -> None:
        completed = run_web(
            textwrap.dedent(
                """
                                import { syncMiniMaxReferenceInputs } from "./custom/web/remote_sockets.js";
                                const node = {
                                  inputs: [], size: [100, 100],
                                  addInput(name, type) { this.inputs.push({ name, type, link: null }); },
                                  removeInput(index) { this.inputs.splice(index, 1); },
                                  computeSize() { return this.size; }, setSize() {},
                                };
                                syncMiniMaxReferenceInputs(node);
                                node.inputs.find((input) => input.name === "ref_image_0").link = 11;
                                syncMiniMaxReferenceInputs(node);
                                process.stdout.write(JSON.stringify(node.inputs.map((input) => input.name)));
                                """
            ),
            app="export const app = { registerExtension() {}, graph: { setDirtyCanvas() {} } };",
            api="export const api = {};",
        )

        names = json.loads(completed.stdout)
        self.assertEqual(
            [
                "ref_image_0",
                "ref_image_1",
                "ref_video_0",
                "ref_video_audio_0",
                "ref_audio_0",
            ],
            names,
        )

    def test_unknown_or_other_peer_profile_does_not_remove_saved_media_links(self):
        run_web('''
            import assert from "node:assert/strict";
            import { syncPromptMediaInputs } from "./custom/web/remote_sockets.js";
            for (const unknown of [true, false]) {
                const inputs = [{ name: "image_0", link: 42 }, { name: "video_0", link: 43 }];
                const node = { inputs, widgets: [{ name: "profile_id", value: "saved" },
                    { name: "server_url", value: "http://other:8188" }],
                    _remotePromptProfiles: unknown ? [] : [{ id: "saved", accepted_media: ["image"] }],
                    _remotePromptProfilesServerUrl: "http://original:8188",
                    removeInput() { assert.fail("Unverified policy must not change graph"); },
                    addInput() { assert.fail("Unverified policy must not change graph"); } };
                syncPromptMediaInputs(node);
                assert.deepEqual(node.inputs, inputs);
            }
        ''')

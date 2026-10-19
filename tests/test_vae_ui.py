"""Public VAE input contracts and shipped browser extension behavior."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest

from remoteworker.vae import schema, service, worker, shared_video
from web_harness import run_web


class VaeSchemaTests(unittest.TestCase):
    def test_remote_model_choices_allow_discovered_names_and_reject_blank(self):
        for cls in service.NODE_CLASS_MAPPINGS.values():
            self.assertTrue(cls.VALIDATE_INPUTS("future/vae.safetensors"))
            self.assertIsInstance(cls.VALIDATE_INPUTS(""), str)
            self.assertEqual([""], cls.INPUT_TYPES()["required"]["vae_name"][0])

    def test_semantic_warmup_signal_and_advanced_inputs(self):
        self.assertEqual(schema.WARMUP_TYPE, service.RemoteVAEWarmup.RETURN_TYPES[0])
        for cls in (service.RemoteVAEDecode, service.RemoteVAEDecodeAudio):
            inputs = cls.INPUT_TYPES()
            self.assertEqual(schema.WARMUP_TYPE, inputs["optional"]["warmup"][0])
            self.assertTrue(inputs["required"]["timeout_seconds"][1]["advanced"])
            self.assertTrue(inputs["required"]["memory_policy"][1]["advanced"])
        self.assertTrue(service.RemoteVAEDecode.INPUT_TYPES()["required"]["tile_size"][1]["advanced"])
        self.assertNotIn("tile_size", service.RemoteVAEDecodeAudio.INPUT_TYPES()["required"])

    def test_internal_workers_remain_executable_without_polluting_node_picker(self):
        self.assertTrue(all(cls.DEV_ONLY for cls in worker.NODE_CLASS_MAPPINGS.values()))

    def test_korean_translations_match_actual_input_names(self):
        root = Path(__file__).resolve().parents[1]
        translations = json.loads((root / "locales/ko/nodeDefs.json").read_text())
        public = service.NODE_CLASS_MAPPINGS | {name: cls for name, cls in shared_video.NODE_CLASS_MAPPINGS.items() if not getattr(cls, "DEV_ONLY", False)}
        for name, cls in public.items():
            inputs = {name for group in cls.INPUT_TYPES().values() for name in group}
            self.assertEqual(inputs, translations[name]["inputs"].keys())
            self.assertEqual(len(cls.RETURN_TYPES), len(translations[name]["outputs"]))

    def test_wrong_media_vae_is_rejected_before_native_decode(self):
        for kind, channels in (("audio", 3), ("image", 2)):
            with self.assertRaisesRegex(ValueError, "VAE"):
                worker.decode(type("Vae", (), {"output_channels": channels})(), {}, kind,
                    False, 512, 64, 64, 8)



@unittest.skipUnless(shutil.which("node"), "Node.js required")
class VaeBrowserContractTests(unittest.TestCase):
    def test_model_refresh_never_changes_selection_and_setup_is_scoped(self):
        run_web('''
            import assert from "node:assert/strict";
            import {applyVaeModels, modelStatus, setupVaeNode, normalizeVaeSockets} from "./custom/web/remote_vae.js";
            const node = {widgets:[{name:"vae_name",value:"chosen.vae",options:{}}]};
            applyVaeModels(node,{vae_name:["different.vae"]});
            assert.equal(node.widgets[0].value,"chosen.vae");
            assert.match(modelStatus(node),/선택한 VAE 없음/);
            node.widgets[0].value="";
            applyVaeModels(node,{vae_name:["first.vae","second.vae"]});
            assert.equal(node.widgets[0].value,"");
            assert.match(modelStatus(node),/선택 필요/);
            const unrelated = {type:"Other",inputs:[{name:"warmup",type:"STRING"}]};
            setupVaeNode(unrelated); normalizeVaeSockets(unrelated);
            assert.equal(unrelated.inputs[0].type,"STRING");
            const {recordVaeDefinition}=await import("./custom/web/vae_definitions.js");
            recordVaeDefinition({name:"RemoteVAEWarmup",output:["REMOTE_VAE_WARMUP","*"]});
            const warmup = {type:"RemoteVAEWarmup",outputs:[{type:"STRING"}]};
            normalizeVaeSockets(warmup);
            assert.equal(warmup.outputs[0].type,"REMOTE_VAE_WARMUP");
        ''')

    def test_refresh_failure_is_short_retryable_and_preserves_model(self):
        run_web('''
            import assert from "node:assert/strict";
            import {app} from "./scripts/app.js";
            import {refreshVaeModels} from "./custom/web/remote_vae.js";
            const node={widgets:[{name:"vae_name",value:"keep.vae"},{name:"server_url",value:"http://peer"},{name:"timeout_seconds",value:30}]};
            const button={name:"refresh"};
            await refreshVaeModels(node,button);
            assert.equal(button.label,"연결 실패 · 다시 조회");
            assert.equal(node.widgets[0].value,"keep.vae");
            assert.equal(app.notices.length,1);
            assert.ok(app.notices[0].detail.length<240);
        ''', app='export const app = {registerExtension(){},notices:[],graph:{setDirtyCanvas(){}},extensionManager:{toast:{add(message){app.notices.push(message);}}}};',
        api='export const api = {async fetchApi(){return {ok:false,async json(){return {error:{message:"x".repeat(800)}};}};}};')

    def test_changed_peer_drops_old_response_without_mutating_options(self):
        run_web('''
            import assert from "node:assert/strict";
            import {api} from "./scripts/api.js";
            import {refreshVaeModels} from "./custom/web/remote_vae.js";
            const node={widgets:[{name:"vae_name",value:"keep.vae",options:{}},{name:"server_url",value:"http://first"},{name:"timeout_seconds",value:30}]};
            const button={name:"refresh"};
            let resolve;
            api.fetchApi=()=>new Promise(r=>resolve=r);
            const request=refreshVaeModels(node,button);
            node.widgets[1].value="http://second";
            resolve({ok:true,async json(){return {vae_name:["wrong.vae"]};}});
            await request;
            assert.equal(node.widgets[0].value,"keep.vae");
            assert.equal(node._remoteVaeModels,undefined);
            assert.match(button.label,/서버 변경됨/);
        ''', api='export const api = {};')


@unittest.skipUnless(shutil.which("node"), "Node.js required")
class VaeDefinitionTests(unittest.TestCase):
    def test_advanced_metadata_reaches_classic_widgets_without_field_duplication(self):
        run_web('''
            import assert from "node:assert/strict";
            import {recordVaeDefinition,applyAdvancedInputs} from "./custom/web/vae_definitions.js";
            recordVaeDefinition({name:"RemoteVAEDecode",input:{required:{timeout_seconds:["INT",{advanced:true}],vae_name:[[""],{}]}}});
            const node={type:"RemoteVAEDecode",widgets:[{name:"timeout_seconds"},{name:"vae_name"}]};
            applyAdvancedInputs(node);
            assert.equal(node.widgets[0].advanced,true);
            assert.equal(node.widgets[0].hidden,true);
            node.showAdvanced=true; applyAdvancedInputs(node);
            assert.equal(node.widgets[0].hidden,false);
            assert.equal(node.widgets[1].advanced,undefined);
        ''')

    def test_model_options_exist_before_saved_workflow_validation(self):
        run_web('''
            import assert from "node:assert/strict";
            import {preloadVaeDefinitions} from "./custom/web/vae_definitions.js";
            const defs={RemoteVAEDecode:{name:"RemoteVAEDecode",input:{required:{vae_name:[[""],{default:""}],server_url:["STRING",{default:"http://peer"}]}}}};
            await preloadVaeDefinitions(defs);
            assert.deepEqual(defs.RemoteVAEDecode.input.required.vae_name[0],["","saved.vae"]);
            assert.equal(defs.RemoteVAEDecode.input.required.vae_name[1].default,"");
        ''',api='export const api={async fetchApi(){return {ok:true,async json(){return {vae_name:["saved.vae"]};}};}};')

    def test_button_identity_and_values_survive_refresh_toggle_and_restore(self):
        run_web('''
            import assert from "node:assert/strict";
            import {setupVaeNode} from "./custom/web/remote_vae.js";
            import {recordVaeDefinition} from "./custom/web/vae_definitions.js";
            recordVaeDefinition({name:"RemoteVAEDecodeAudio",input:{required:{timeout_seconds:["INT",{advanced:true}]}}});
            const node={type:"RemoteVAEDecodeAudio",size:[430,180],widgets:[
                {name:"vae_name",value:"chosen.vae",options:{}},{name:"server_url",value:"http://peer"},{name:"timeout_seconds",value:180}],
                addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;},
                computeSize(){return [430,this.showAdvanced?200:140];},setSize(value){this.size=value;}};
            setupVaeNode(node);
            await Promise.resolve(); await Promise.resolve(); await new Promise(r=>setImmediate(r));
            assert.equal(node._remoteVaeRefresh.name,"remote_vae_models");
            assert.match(node._remoteVaeRefresh.label,/모델 1개/);
            assert.equal(node._remoteVaeDetails.name,"remote_vae_details");
            node._remoteVaeDetails.callback();
            assert.equal(node.showAdvanced,true);
            assert.equal(node.widgets[2].value,180);
            assert.equal(node.widgets[2].hidden,false);
            setupVaeNode(node);
            assert.equal(node.widgets.length,5);
            assert.equal(node._remoteVaeDetails.label,"고급 설정 숨기기");
            assert.equal(node._remoteVaeRefresh.serialize,false);
            assert.equal(node._remoteVaeDetails.serialize,false);
        ''', api='export const api={async fetchApi(){return {ok:true,async json(){return {vae_name:["chosen.vae"]};}};}};')

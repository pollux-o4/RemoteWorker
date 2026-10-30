"""Model discovery, recovery and node-local upload interaction contracts."""
import shutil
import unittest

from web_harness import run_web


@unittest.skipUnless(shutil.which("node"), "Node.js required")
class ModelControlUX(unittest.TestCase):
    def test_discovery_preserves_values_and_reselection_clears_warning(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {refreshRemoteOptions, applyRemoteOptions} from './custom/web/remote_choices.js';
            const widgets=[{name:'server_url',value:'http://peer'},{name:'timeout_seconds',value:30},
                {name:'clip_name',value:'old.clip',options:{}},{name:'clip_type',value:'old-type',options:{}},{name:'device',value:'old-device',options:{}}];
            const node={widgets}; const button={};
            await refreshRemoteOptions(node,button);
            assert.deepEqual(widgets.slice(2).map(w=>w.value),['old.clip','old-type','old-device']);
            assert.match(button.name,/다시 선택/);
            assert.ok(button.name.length<=24);
            for (const [i,value] of ['new.clip','new-type','cpu'].entries()) {
                const w=widgets[i+2]; assert.ok(w.options.values.includes(w.value));
                w.value=value; w.callback(value);
            }
            assert.match(button.name,/CLIP 1개/);
            assert.doesNotMatch(widgets[2].options.tooltip,/미확인/);
            const empty={widgets:['clip_name','clip_type','device'].map(name=>({name,value:'',options:{}}))};
            applyRemoteOptions(empty,{clip_name:['new.clip'],clip_type:['new-type'],device:['cpu']});
            assert.deepEqual(empty.widgets.map(w=>w.value),['','','']);
        ''', api="export const api={async fetchApi(){return {ok:true,async json(){return {clip_name:['new.clip'],clip_type:['new-type'],device:['cpu']};}};}};")

    def test_same_peer_double_click_keeps_one_pending_lookup(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {api} from './scripts/api.js';
            import {refreshPeerChoices} from './custom/web/remote_qwen_image21.js';
            const node={widgets:[{name:'remote_text_encoder',value:'chosen',options:{}},{name:'server_url',value:'http://peer'},{name:'timeout_seconds',value:30}]};
            let resolve,calls=0;api.fetchApi=()=>{calls++;return new Promise(r=>resolve=r);};
            const button={};const first=refreshPeerChoices(node,button,'encoder');
            await refreshPeerChoices(node,button,'encoder');
            assert.equal(calls,1);assert.match(button.name,/조회 중/);assert.equal(button._remoteQwenRefreshInFlight,true);
            resolve({ok:true,async json(){return {clip_name:['chosen']};}});await first;
            assert.equal(button._remoteQwenRefreshInFlight,false);assert.match(button.name,/1개/);
        ''', api="export const api={};")

    def test_new_peer_retry_supersedes_old_request_and_ignores_old_failure(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {app} from './scripts/app.js';
            import {api} from './scripts/api.js';
            import {refreshPeerChoices} from './custom/web/remote_qwen_image21.js';
            const target={name:'remote_text_encoder',value:'chosen',options:{}};
            const server={name:'server_url',value:'http://a'};
            const node={widgets:[target,server,{name:'timeout_seconds',value:30}]};
            const pending=[];api.fetchApi=(_,options)=>new Promise((resolve,reject)=>pending.push({resolve,reject,server:JSON.parse(options.body).server_url}));
            const button={};const first=refreshPeerChoices(node,button,'encoder');
            server.value='http://b';server.callback();
            const second=refreshPeerChoices(node,button,'encoder');
            assert.deepEqual(pending.map(p=>p.server),['http://a','http://b']);
            pending[1].resolve({ok:true,async json(){return {clip_name:['chosen','second']};}});
            await second;const label=button.name;
            assert.match(label,/2개/);
            pending[0].reject(new Error('old peer failed'));await first;
            assert.equal(button.name,label);assert.equal(target.value,'chosen');
            assert.deepEqual(target._remotePeerChoices,['chosen','second']);
            assert.equal(app.notices.length,0);assert.equal(button._remoteQwenRefreshInFlight,false);
        ''', app="export const app={registerExtension(){},notices:[],graph:{setDirtyCanvas(){}},extensionManager:{toast:{add(value){app.notices.push(value)}}}};", api="export const api={};")

    def test_peer_roundtrip_invalidates_old_success_and_old_error(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {app} from './scripts/app.js';
            import {api} from './scripts/api.js';
            import {refreshPeerChoices} from './custom/web/remote_qwen_image21.js';
            for (const fail of [false,true]) {
                const server={name:'server_url',value:'http://a'};
                const target={name:'vae_name',value:'chosen',options:{values:['chosen']}};
                const node={widgets:[server,target,{name:'timeout_seconds',value:30}]};
                let resolve,reject;api.fetchApi=()=>new Promise((yes,no)=>{resolve=yes;reject=no;});
                const button={};const request=refreshPeerChoices(node,button,'vae');
                server.value='http://b';server.callback();server.value='http://a';server.callback();
                if (fail) reject(new Error('obsolete')); else resolve({ok:true,async json(){return {vae_name:['wrong']};}});
                await request;
                assert.equal(target.value,'chosen');assert.deepEqual(target.options.values,['chosen']);
                assert.match(button.name,/서버 변경됨/);assert.equal(button._remoteQwenRefreshInFlight,false);
            }
            assert.equal(app.notices.length,0);
        ''', app="export const app={registerExtension(){},notices:[],graph:{setDirtyCanvas(){}},extensionManager:{toast:{add(value){app.notices.push(value)}}}};", api="export const api={};")

    def test_failure_is_short_visible_and_retryable_with_full_detail_available(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {app} from './scripts/app.js';
            import {api} from './scripts/api.js';
            import {refreshPeerChoices} from './custom/web/remote_qwen_image21.js';
            const target={name:'vae_name',value:'chosen',options:{}};
            const node={widgets:[target,{name:'server_url',value:'http://peer'},{name:'timeout_seconds',value:30}]};
            const detail='server unavailable '.repeat(40);let fail=true;
            api.fetchApi=async()=>fail?{ok:false,status:503,async json(){return {error:detail};}}:{ok:true,async json(){return {vae_name:['chosen']};}};
            const button={};await refreshPeerChoices(node,button,'vae');
            assert.ok(button.name.length<=24);assert.match(button.name,/조회 실패.*재시도/);
            assert.ok(button.options.tooltip.includes(detail));assert.equal(target.value,'chosen');
            assert.equal(app.notices.length,1);assert.ok(app.notices[0].detail.length<260);
            assert.match(app.notices[0].detail,/서버 주소.*재시도/);
            fail=false;await refreshPeerChoices(node,button,'vae');
            assert.match(button.name,/1개/);assert.equal(target.value,'chosen');
        ''', app="export const app={registerExtension(){},notices:[],graph:{setDirtyCanvas(){}},extensionManager:{toast:{add(value){app.notices.push(value)}}}};", api="export const api={};")

    def test_narrow_vae_node_keeps_width_values_and_button_identity(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {recordVaeDefinition} from './custom/web/vae_definitions.js';
            import {setupVaeNode} from './custom/web/remote_vae.js';
            recordVaeDefinition({name:'RemoteVAEDecodeAudio',input:{required:{timeout_seconds:['INT',{advanced:true}]}}});
            const node={type:'RemoteVAEDecodeAudio',size:[220,180],widgets:[{name:'vae_name',value:'chosen',options:{}},{name:'server_url',value:'http://peer'},{name:'timeout_seconds',value:123}],
                addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;},computeSize(){return [360,this.showAdvanced?220:160]},setSize(size){this.size=size}};
            setupVaeNode(node);await new Promise(r=>setImmediate(r));
            const controls=[node._remoteVaeRefresh,node._remoteVaeDetails];
            assert.equal(node.size[0],220);
            controls[1].callback();assert.equal(node.widgets[2].hidden,false);assert.equal(node.widgets[2].value,123);
            controls[1].callback();assert.equal(node.widgets[2].hidden,true);assert.equal(node.size[0],220);
            setupVaeNode(node);assert.deepEqual([node._remoteVaeRefresh,node._remoteVaeDetails],controls);
            assert.equal(node.widgets.length,5);assert.ok(controls.every(w=>w.serialize===false&&w.label.length<=24));
        ''', api="export const api={async fetchApi(){return {ok:true,async json(){return {vae_name:['chosen']};}};}};")

    def test_model_definition_lookup_preserves_explicit_defaults(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {discoverMiniMaxModelDefinitions} from './custom/web/remote_model_definitions.js';
            import {preloadVaeDefinitions} from './custom/web/vae_definitions.js';
            const fields={clip_name:[['old/model'],{default:'old/model'}],vae_name:[['old.vae'],{default:'old.vae'}],server_url:['STRING',{default:'http://peer'}]};
            const defs={RemoteMiniMaxH3ImageToVideo:{name:'RemoteMiniMaxH3ImageToVideo',input:{required:fields}}};
            await discoverMiniMaxModelDefinitions(defs);
            assert.equal(fields.clip_name[1].default,'old/model');assert.ok(fields.clip_name[0].includes('old/model'));
            assert.equal(fields.vae_name[1].default,'old.vae');assert.ok(fields.vae_name[0].includes('old.vae'));
            const vae={RemoteVAEDecode:{name:'RemoteVAEDecode',input:{required:{vae_name:[['configured'],{default:'configured'}],server_url:['STRING',{default:'http://peer'}]}}}};
            await preloadVaeDefinitions(vae);
            assert.equal(vae.RemoteVAEDecode.input.required.vae_name[1].default,'configured');
            assert.ok(vae.RemoteVAEDecode.input.required.vae_name[0].includes('configured'));
        ''', api="export const api={async fetchApi(path){return {ok:true,async json(){return path.includes('clip-')?{clip_name:['new/model']}:{vae_name:['new.vae']};}};}};")

    def test_maximum_reference_socket_keeps_its_link_and_object(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {syncMiniMaxReferenceInputs} from './custom/web/remote_sockets.js';
            for (const [prefix,maximum] of [['ref_image',9],['ref_video',3],['ref_video_audio',3],['ref_audio',3]]) {
                const last={name:`${prefix}_${maximum-1}`,type:'IMAGE',link:99};
                const node={size:[220,160],inputs:Array.from({length:maximum},(_,i)=>i===maximum-1?last:{name:`${prefix}_${i}`,link:null}),
                    addInput(name,type){this.inputs.push({name,type,link:null});},removeInput(i){this.inputs.splice(i,1);},computeSize(){return this.size},setSize(){}};
                syncMiniMaxReferenceInputs(node);syncMiniMaxReferenceInputs(node);
                assert.equal(node.inputs.find(input=>input.name===last.name),last);assert.equal(last.link,99);
                assert.equal(node.inputs.filter(input=>input.name.startsWith(prefix+'_')&&/^[0-9]+$/.test(input.name.slice(prefix.length+1))).length,maximum);
            }
        ''')

    def test_autogrow_restores_group_order_without_recreating_inputs_or_fixed_slots(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {syncMiniMaxReferenceInputs} from './custom/web/remote_sockets.js';
            const image={name:'ref_image_0',type:'IMAGE',link:1};
            const fixed={name:'vae_name_override',type:'*',link:2};
            const video={name:'ref_video_0',type:'IMAGE',link:3};
            const links=new Map([[1,{target_slot:0}],[2,{target_slot:1}],[3,{target_slot:2}]]);
            const node={size:[220,160],inputs:[image,fixed,video],graph:{links},addInput(name,type){this.inputs.push({name,type,link:null});},removeInput(i){this.inputs.splice(i,1)},computeSize(){return this.size},setSize(){}};
            syncMiniMaxReferenceInputs(node);syncMiniMaxReferenceInputs(node);
            assert.equal(node.inputs[0],image);assert.equal(node.inputs[1],fixed);
            assert.equal(node.inputs.find(input=>input.name==='ref_video_0'),video);
            assert.deepEqual(node.inputs.map(input=>input.name),['ref_image_0','vae_name_override','ref_image_1','ref_video_0','ref_video_1','ref_video_audio_0','ref_audio_0']);
            for (const [id,link] of links) assert.equal(node.inputs[link.target_slot].link,id);
        ''')

    def test_upload_failure_preserves_selection_and_retry_uses_unique_server_name(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {app} from './scripts/app.js';
            import {api} from './scripts/api.js';
            import {uploadCond} from './custom/web/uploadcond.js';
            globalThis.FormData=class {values=new Map();append(name,value){this.values.set(name,value)}};
            const selected={name:'filename',value:'kept.cond',options:{values:['kept.cond']},callback(){selected.called=true;}};
            const button={name:'conditioning_upload',options:{serialize:false}};const node={widgets:[selected],_conditioningUpload:button};
            let calls=0;api.fetchApi=async(path,{body})=>{calls++;assert.equal(body.values.get('overwrite'),'false');return calls===1?{ok:false,status:500,async json(){return {error:'disk unavailable'};}}:{ok:true,async json(){return {name:'retry_1.cond',subfolder:'uploads'};}};};
            assert.equal(await uploadCond({name:'retry.cond'},node),false);
            assert.equal(selected.value,'kept.cond');assert.equal(selected.called,undefined);
            assert.match(button.label,/실패.*다시 선택/);assert.equal(app.notices.length,1);
            assert.equal(await uploadCond({name:'retry.cond'},node),true);
            assert.equal(selected.value,'uploads/retry_1.cond');assert.ok(selected.options.values.includes('kept.cond'));
            assert.equal(button.name,'conditioning_upload');assert.ok(button.label.length<=24);
        ''', app="export const app={registerExtension(){},notices:[],graph:{setDirtyCanvas(){}},extensionManager:{toast:{add(value){app.notices.push(value)}}}};", api="export const api={};")

    def test_uploaded_file_is_not_reported_as_failed_when_selection_callback_throws(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {app} from './scripts/app.js';
            import {uploadCond} from './custom/web/uploadcond.js';
            globalThis.FormData=class {append(){}};
            const selected={name:'filename',value:'kept.cond',options:{values:['kept.cond']},callback(){throw new Error('widget unavailable')}};
            const button={options:{}};const node={widgets:[selected],_conditioningUpload:button};
            assert.equal(await uploadCond({name:'new.cond'},node),true);
            assert.equal(selected.value,'new.cond');assert.match(button.label,/업로드 완료/);
            assert.equal(app.notices[0].severity,'warn');assert.match(app.notices[0].detail,/파일 저장 완료/);
        ''', app="export const app={registerExtension(){},notices:[],graph:{setDirtyCanvas(){}},extensionManager:{toast:{add(value){app.notices.push(value)}}}};", api="export const api={async fetchApi(){return {ok:true,async json(){return {name:'new.cond'};}};}};")

    def test_file_picker_cancel_and_upload_controls_are_ephemeral(self):
        run_web('''
            import assert from 'node:assert/strict';
            import {app} from './scripts/app.js';
            import {pickAndUpload} from './custom/web/uploadcond.js';
            let input;globalThis.document={createElement(){input={style:{},click(){this.clicked=true},remove(){this.removed=true}};return input},body:{append(){}}};
            pickAndUpload({});assert.equal(input.clicked,true);input.oncancel();assert.equal(input.removed,true);
            class Node{widgets=[{name:'filename',value:'kept.cond'}];addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;}}
            await app.extension.beforeRegisterNodeDef(Node,{name:'LoadConditioning'});const node=new Node();node.onNodeCreated();
            assert.equal(node._conditioningUpload.label,'.cond 업로드');assert.equal(node._conditioningUpload.serialize,false);
            assert.equal(node._conditioningUpload.options.serialize,false);
        ''', app="export const app={registerExtension(extension){this.extension=extension},graph:{setDirtyCanvas(){}}};")

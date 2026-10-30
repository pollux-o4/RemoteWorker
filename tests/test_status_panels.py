"""Status-panel recovery and completion boundaries through shipped browser modules."""
import shutil
import unittest
from pathlib import Path
from web_harness import run_web

DOM = (Path(__file__).parent / "fixtures/status_dom.mjs").read_text().replace("export ", "")


@unittest.skipUnless(shutil.which("node"), "Node.js required")
class StatusPanelTests(unittest.TestCase):
    def run_dom(self, source, **kwargs):
        return run_web(DOM + "\ninstallDOM();\n" + source, **kwargs)

    def test_ready_means_reentry_not_decode_success_and_full_job_identity_survives(self):
        run_web('''
          import assert from 'node:assert/strict';
          import {schedulerText} from './custom/web/scheduler_view.js';
          const state=schedulerText({}, {enabled:true,jobs:[{prompt_id:'original-job-A',phase:'ready'},
            {prompt_id:'original-job-B',phase:'restoring'},{prompt_id:'original-job-C',phase:'failed'}]});
          assert.equal(state.title,'작업 중단');
          assert.equal(state.jobs[0].id,'original-job-A');
          assert.match(state.jobs[0].label,/복귀 대기/);
          assert.doesNotMatch(state.timing,/완료|로컬 처리/);
          assert.match(state.jobs[1].label,/복원/);
          assert.match(state.jobs[2].label,/중단/);
          const preview=schedulerText({}, {enabled:true,jobs:[{prompt_id:'A',phase:'ready',preview_nodes:['p']}]});
          assert.match(preview.jobs[0].label,/미리보기/);
        ''')

    def test_preparation_dialog_single_status_details_and_close_abort_focus(self):
        self.run_dom('''
          import assert from 'node:assert/strict';
          const {api}=await import('./scripts/api.js');
          const {showPreparation}=await import('./custom/web/preparation.js');
          const opener=document.createElement('button');document.body.append(opener);opener.focus();
          let finish, signal;
          api.fetchApi=(_,options)=>{signal=options.signal;return new Promise(resolve=>finish=resolve);};
          const request=showPreparation();
          const dialog=document.querySelector('dialog.remote-preparation');
          assert.equal(document.activeElement.textContent,'닫기');
          dialog.querySelector('button').click();
          assert.equal(signal.aborted,true);assert.equal(document.activeElement,opener);
          finish({ok:true,json:async()=>({pipeline:{enabled:true,jobs:[]},events:[]})});await request;
          assert.equal(document.querySelector('dialog.remote-preparation'),null);
          api.fetchApi=async()=>({ok:true,json:async()=>({pipeline:{enabled:true,local_prompt_id:'job-B-full-id',jobs:[
            {prompt_id:'job-A-full-id',phase:'remote'},{prompt_id:'job-B-full-id',phase:'local'}]},events:[]})});
          await showPreparation();
          const current=document.querySelector('dialog.remote-preparation');
          assert.equal(current.querySelectorAll('[role=status]').length,1);
          assert.equal(current.querySelector('details').open,undefined);
          assert.equal(current.querySelector('[role=status]').textContent,'로컬·원격 병행');
          assert.equal(current.querySelector('.preparation-schedule').textContent.includes('병행'),false);
          assert.match(current.querySelector('details').textContent,/job-A-full-id/);
          current.querySelector('button').click();
        ''',api='export const api={};')

    def test_delivery_cancel_server_roundtrip_retry_and_intrinsic_height(self):
        self.run_dom('''
          import assert from 'node:assert/strict';
          const {api}=await import('./scripts/api.js');
          const {setupDelivery,checkDelivery}=await import('./custom/web/vae_delivery.js');
          const server={name:'server_url',value:'http://A'};
          const mode={name:'output_mode',value:'shared'};
          let widget;
          const node={type:'RemoteVAEDecode',widgets:[server,mode],size:[260,200],outputs:[{},{}],
            addDOMWidget(_,__,panel){document.body.append(panel);widget={};return widget;},
            computeSize(){return [this.size[0],widget?.computeSize?.(this.size[0])[1]||100];},setSize(size){this.size=size;}};
          setupDelivery(node);
          const panel=node._deliveryPanel,status=panel.querySelector('[role=status]'),button=panel.querySelector('button');
          panel._height=214;assert.equal(widget.computeSize(260)[1],214);
          assert.equal(node.size[0],260);
          const pending=[];
          api.fetchApi=(_,options)=>new Promise(resolve=>pending.push({resolve,signal:options.signal}));
          const first=checkDelivery(node);assert.equal(button.textContent,'확인 취소');
          server.value='http://B';server.callback();server.value='http://A';server.callback();
          pending[0].resolve({ok:true,json:async()=>({root:'/stale-A'})});await first;
          assert.equal(status.textContent,'공유 폴더 확인 필요');
          assert.equal(pending[0].signal.aborted,true);
          const cancelled=checkDelivery(node);button.click();
          assert.equal(status.textContent,'확인 취소됨');assert.equal(pending[1].signal.aborted,true);
          pending[1].resolve({ok:true,json:async()=>({root:'/cancelled'})});await cancelled;
          assert.equal(status.textContent,'확인 취소됨');
          api.fetchApi=async()=>({ok:false,json:async()=>({error:'permission denied on peer'})});
          await checkDelivery(node);assert.equal(button.textContent,'다시 확인');
          assert.match(panel.querySelector('details').textContent,/permission denied/);
          const diagnostic=panel.querySelector('[data-diagnostic]');
          assert.equal(diagnostic.getAttribute('tabindex'),'0');
          assert.equal(diagnostic.getAttribute('role'),'region');
          assert.equal(diagnostic.getAttribute('aria-label'),'공유 폴더 연결 상세');
          diagnostic.focus();assert.equal(document.activeElement,diagnostic);
          assert.equal(status.title,'');
          api.fetchApi=async()=>({ok:true,json:async()=>({root:'/shared',ready:false})});
          await checkDelivery(node);assert.equal(status.dataset.state,'error');
          api.fetchApi=async()=>({ok:true,json:async()=>({})});
          await checkDelivery(node);assert.equal(status.dataset.state,'error');
          api.fetchApi=async()=>({ok:true,json:async()=>({root:'/shared',ready:true})});
          await checkDelivery(node);assert.equal(status.textContent,'공유 폴더 연결됨');
          node.onExecuted({shared_transfer:[]});assert.equal(status.textContent,'공유 폴더 연결됨');
          node.onExecuted({shared_transfer:['image']});assert.equal(status.textContent,'전송 완료');
          assert.match(panel.querySelector('[data-next]').textContent,/후속 노드/);
          mode.value='tensor';mode.callback();assert.equal(panel.hidden,true);
          node.onRemoved();
        ''',api='export const api={};')

    def test_connection_and_status_timeouts_expose_retry_without_false_success(self):
        self.run_dom('''
          import assert from 'node:assert/strict';
          const {api}=await import('./scripts/api.js');
          const {checkDelivery}=await import('./custom/web/vae_delivery.js');
          const {showPreparation}=await import('./custom/web/preparation.js');
          const timers=[];
          globalThis.setTimeout=(callback,delay)=>{timers.push({callback,delay});return timers.length;};
          globalThis.clearTimeout=()=>{};
          globalThis.setInterval=()=>1;globalThis.clearInterval=()=>{};
          api.fetchApi=(_,options)=>new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(Object.assign(new Error('aborted'),{name:'AbortError'}))));
          const panel=document.createElement('section'),status=document.createElement('span'),button=document.createElement('button');
          status.setAttribute('role','status');panel.append(status,button);
          const node={widgets:[{name:'server_url',value:'http://peer'}],_deliveryPanel:panel};
          const pending=checkDelivery(node);assert.equal(timers[0].delay,10000);timers[0].callback();await pending;
          assert.equal(status.dataset.state,'error');assert.equal(status.textContent,'연결 응답 지연');assert.equal(button.textContent,'다시 확인');
          api.fetchApi=async()=>({ok:true,json:async()=>({root:'/shared',ready:true})});await checkDelivery(node);
          assert.equal(status.textContent,'공유 폴더 연결됨');
          api.fetchApi=(_,options)=>new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(Object.assign(new Error('aborted'),{name:'AbortError'}))));
          const dialogRequest=showPreparation();timers.at(-1).callback();await dialogRequest;
          const dialog=document.querySelector('dialog.remote-preparation');
          assert.match(dialog.querySelector('[role=status]').textContent,/응답 지연/);
          const retry=dialog.querySelectorAll('button').at(-1);assert.equal(retry.disabled,false);assert.equal(retry.textContent,'다시 확인');
          api.fetchApi=async()=>({ok:true,json:async()=>({pipeline:{enabled:true,jobs:[]},events:[]})});
          await retry.onclick();assert.equal(dialog.querySelector('[role=status]').textContent,'대기');
          dialog.querySelector('button').click();
        ''',api='export const api={};')

    def test_connection_requires_explicit_ready_and_nonempty_root_while_stale_bad_payload_is_ignored(self):
        self.run_dom('''
          import assert from 'node:assert/strict';
          const {api}=await import('./scripts/api.js');
          const {setupDelivery,checkDelivery}=await import('./custom/web/vae_delivery.js');
          const node={type:'RemoteVAEDecodeAudio',widgets:[{name:'server_url',value:'http://peer'},{name:'output_mode',value:'shared'}],
            addDOMWidget(_,__,panel){document.body.append(panel);return {};}};
          setupDelivery(node);const status=node._deliveryPanel.querySelector('[role=status]');
          for (const ready of [undefined,null,false,0,1,'false','true',{},[]]) {
            api.fetchApi=async()=>({ok:true,json:async()=>({root:'/shared',ready})});
            await checkDelivery(node);assert.equal(status.dataset.state,'error',String(ready));
            assert.notEqual(status.textContent,'공유 폴더 연결됨');
          }
          for (const root of ['', '  ', null, 42, {}]) {
            api.fetchApi=async()=>({ok:true,json:async()=>({root,ready:true})});
            await checkDelivery(node);assert.equal(status.dataset.state,'error',String(root));
          }
          let finish;
          api.fetchApi=()=>new Promise(resolve=>finish=resolve);
          const stale=checkDelivery(node);
          api.fetchApi=async()=>({ok:true,json:async()=>({root:'/confirmed',ready:true})});
          await checkDelivery(node);assert.equal(status.dataset.state,'ready');
          finish({ok:true,json:async()=>({root:'/stale',ready:'false'})});await stale;
          assert.equal(status.dataset.state,'ready');assert.equal(status.textContent,'공유 폴더 연결됨');
          assert.equal(node._deliveryPanel.querySelector('[data-diagnostic]').textContent,'/confirmed');
        ''',api='export const api={};')

    def test_removed_node_ignores_delivery_response_and_preserves_callbacks(self):
        self.run_dom('''
          import assert from 'node:assert/strict';
          const {api}=await import('./scripts/api.js');
          const {setupDelivery,checkDelivery}=await import('./custom/web/vae_delivery.js');
          let removed=0,resolve;
          const node={type:'RemoteVAEDecodeAudio',widgets:[{name:'server_url',value:'http://peer'},{name:'output_mode',value:'shared'}],
            addDOMWidget(_,__,panel){document.body.append(panel);return {};},onRemoved(){removed++;}};
          setupDelivery(node);const status=node._deliveryPanel.querySelector('[role=status]');
          api.fetchApi=()=>new Promise(done=>resolve=done);const check=checkDelivery(node);
          node.onRemoved();resolve({ok:true,json:async()=>({root:'/late'})});await check;
          assert.equal(removed,1);assert.notEqual(status.textContent,'공유 폴더 연결됨');
        ''',api='export const api={};')

    def test_final_video_and_cleanup_evidence_are_distinct_and_height_tracks_details(self):
        self.run_dom('''
          import assert from 'node:assert/strict';
          const {setupResult}=await import('./custom/web/shared_combine.js');
          let widget,panel;
          const node={type:'RemoteVAECombineMedia',size:[260,100],addDOMWidget(_,__,element){panel=element;document.body.append(panel);widget={};return widget;},
            computeSize(){return [260,widget?.computeSize?.(260)[1]||60];},setSize(size){this.size=size;}};
          setupResult(node);const url='/remote-vae/media/'+'a'.repeat(32);
          node.onExecuted({shared_media:[{final:false,url}],source_cleanup:[true]});
          assert.equal(panel.querySelector('video'),null);assert.equal(document.events.length,0);
          node.onExecuted({shared_media:[{final:true,url}],source_cleanup:[false]});
          assert.equal(panel.querySelector('[role=status]').textContent,'영상 완료');
          assert.match(panel.querySelector('details').textContent,/보존/);assert.equal(document.events.length,0);
          node.onExecuted({shared_media:[{final:true,url}]});
          assert.match(panel.querySelector('details').textContent,/상태 확인 필요/);assert.equal(document.events.length,0);
          const oldVideo=panel.querySelector('video');
          node.onExecuted({shared_media:[{final:true,url}],source_cleanup:['true']});
          assert.equal(oldVideo.paused,true);assert.equal(document.events.length,0);
          oldVideo.dispatchEvent({type:'error'});assert.equal(panel.querySelector('[role=status]').textContent,'영상 완료');
          panel.querySelector('video').dispatchEvent({type:'error'});
          assert.equal(panel.querySelector('[role=status]').textContent,'미리보기 로드 실패');
          assert.equal(panel.querySelector('a').textContent,'영상 다시 열기');
          node.onExecuted({shared_media:[{final:true,url}],source_cleanup:[true]});
          assert.equal(document.events.length,1);assert.match(panel.querySelector('details').textContent,/정리됨/);
          panel._height=370;assert.equal(widget.computeSize(260)[1],370);
          assert.equal(node.size[0],260);node.onRemoved();assert.equal(panel.querySelector('video').paused,true);
        ''',api='export const api={apiURL:path=>path};')

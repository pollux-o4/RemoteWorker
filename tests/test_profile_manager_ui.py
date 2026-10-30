"""Profile manager state/recovery contracts with mocked DOM and HTTP effects."""

from pathlib import Path
import shutil
import subprocess
import unittest

from web_harness import run_web

FIXTURE = Path(__file__).with_name("profile_manager_dom.mjs")
PRELUDE = r"""
    import assert from "node:assert/strict";
    import {api} from "./scripts/api.js";
    import {openPromptProfileManager, profileMutationObserved} from "./custom/web/remote_profile_dialog.js";
    const {installProfileDom} = await import(process.argv[2]);
    const {role, Element} = installProfileDom();
    const settle = () => new Promise(resolve => setTimeout(resolve, 0));
    const profile = id => ({id, description:`${id} description`, clip_name:"encoder.safetensors",
        clip_type:"minimax", device:"default", max_length:256, system_prompt:`${id} instructions`, accepted_media:["image"]});
    const clips = {clip_name:["encoder.safetensors"],clip_type:["minimax"],device:["default"]};
    const node = {widgets:[{name:"profile_id",value:"alpha",options:{}},
        {name:"server_url",value:"http://peer:8188"},{name:"timeout_seconds",value:30}]};
    const adapter = {normalize:payload=>payload.profiles,apply(node,payload){node.applied=payload;}};
    const response = payload => ({ok:true,json:async()=>payload});
"""


@unittest.skipUnless(shutil.which("node"), "Node.js required")
class ProfileManagerUi(unittest.TestCase):
    def run_contract(self, source):
        try:
            run_web(PRELUDE + source, args=(str(FIXTURE),))
        except subprocess.CalledProcessError as error:
            self.fail(error.stderr)

    def test_selection_refresh_keeps_missing_value_and_manager_sentinels_never_persist(
        self,
    ):
        self.run_contract(r"""
            import {applyPromptProfiles, setupPromptProfileManagerShortcut, PROFILE_MANAGER_OPTION}
                from "./custom/web/remote_profiles.js";
            node.widgets[0].value="missing";
            applyPromptProfiles(node,{profiles:[profile("alpha")]});
            assert.equal(node.widgets[0].value,"missing");
            assert.ok(node.widgets[0].options.values.includes("missing"));
            api.fetchApi=async path=>response(path.includes("clip-loader")?clips:{profiles:[profile("alpha")]});
            for (const sentinel of [PROFILE_MANAGER_OPTION,"Manage profiles..."]) {
                node.widgets[0].callback(sentinel);
                assert.equal(node.widgets[0].value,"missing");
                await settle();
                node._remotePromptManagerDialog.close();
            }
            assert.equal(PROFILE_MANAGER_OPTION,"프로필 관리…");
        """)

    def test_profile_drafts_survive_target_switch_errors_and_close_with_focus_restore(
        self,
    ):
        self.run_contract(r"""
            api.fetchApi=async path=>response(path.includes("clip-loader")?clips:{profiles:[profile("alpha"),profile("beta")]});
            const opener=new Element("button");opener.focus();
            const dialog=await openPromptProfileManager(node,adapter);
            assert.equal(role(dialog,"description").textContent,"alpha description");
            role(dialog,"edit").click();
            role(dialog,"description-input").value="unsaved alpha";
            role(dialog,"picker").value="beta";role(dialog,"picker").dispatch("change");
            role(dialog,"edit").click();
            assert.equal(role(dialog,"description-input").value,"beta description");
            role(dialog,"picker").value="alpha";role(dialog,"picker").dispatch("change");
            role(dialog,"edit").click();
            assert.equal(role(dialog,"description-input").value,"unsaved alpha");
            role(dialog,"close").click();
            assert.equal(document.activeElement,opener);
            const reopened=await openPromptProfileManager(node,adapter);
            assert.equal(role(reopened,"description-input").value,"unsaved alpha");
            assert.match(role(reopened,"status").textContent,/복원/);
            role(reopened,"close").click();
        """)

    def test_confirmed_save_with_failed_refresh_retries_only_read_and_selects_created_profile(
        self,
    ):
        self.run_contract(r"""
            let records=[profile("alpha")], writes=0, failRefresh=false;
            api.fetchApi=async(path,options)=>{
                if(path.endsWith("/manage")){
                    writes++;const payload=JSON.parse(options.body);records.push(payload.profile);failRefresh=true;return response({saved:true});
                }
                if(path.includes("clip-loader"))return response(clips);
                if(failRefresh){failRefresh=false;throw new Error("offline");}
                return response({profiles:records});
            };
            const dialog=await openPromptProfileManager(node,adapter);
            role(dialog,"add").click();
            role(dialog,"id").value="alpha";role(dialog,"description-input").value="new description";
            role(dialog,"system-prompt").value="new instructions";
            role(dialog,"save").click();await settle();
            assert.equal(writes,0);assert.match(role(dialog,"id").validationMessage,/이미/);
            role(dialog,"id").value="created";role(dialog,"save").click();await settle();
            assert.equal(writes,1);assert.equal(dialog.open,true);
            assert.match(role(dialog,"status").textContent,/저장됨/);
            assert.equal(role(dialog,"save").disabled,true);
            role(dialog,"save").click();assert.equal(writes,1);
            role(dialog,"retry").click();await settle();
            assert.equal(writes,1);assert.equal(dialog.open,false);
            assert.equal(node.widgets[0].value,"created");
            assert.ok(node.applied.profiles.some(item=>item.id==="created"));
        """)

    def test_uncertain_commit_is_reconciled_after_reopening_without_replaying_write(
        self,
    ):
        self.run_contract(r"""
            let records=[profile("alpha")],writes=0;
            api.fetchApi=async(path,options)=>{
                if(path.endsWith("/manage")){writes++;records=[JSON.parse(options.body).profile];throw new Error("response lost");}
                return response(path.includes("clip-loader")?clips:{profiles:records});
            };
            const dialog=await openPromptProfileManager(node,adapter);
            role(dialog,"edit").click();role(dialog,"description-input").value="committed but response lost";
            role(dialog,"save").click();await settle();
            assert.equal(writes,1);assert.match(role(dialog,"status").textContent,/미확인/);
            role(dialog,"close").click();
            const reopened=await openPromptProfileManager(node,adapter);
            assert.equal(writes,1);assert.equal(reopened.open,false);
            assert.equal(node.applied.profiles[0].description,"committed but response lost");
        """)

    def test_load_failure_empty_state_and_superseded_peer_never_mutate_node(self):
        self.run_contract(r"""
            let rejectLoad=true;
            api.fetchApi=async path=>{
                if(rejectLoad)throw new Error("server unavailable");
                return response(path.includes("clip-loader")?clips:{profiles:[]});
            };
            const dialog=await openPromptProfileManager(node,adapter);
            assert.match(role(dialog,"status").textContent,/조회 실패/);
            assert.equal(node.widgets[0].value,"alpha");
            rejectLoad=false;role(dialog,"retry").click();await settle();
            assert.match(role(dialog,"description").textContent,/프로필 없음/);
            assert.equal(role(dialog,"add").disabled,false);
            role(dialog,"close").click();
            let resolveProfiles;
            api.fetchApi=path=>path.includes("clip-loader")?Promise.resolve(response(clips)):
                new Promise(resolve=>{resolveProfiles=resolve;});
            const loading=openPromptProfileManager(node,adapter);
            const server=node.widgets[1];server.value="http://other:8188";server.callback();
            server.value="http://peer:8188";server.callback();
            resolveProfiles(response({profiles:[profile("alpha")]}));
            const stale=await loading;
            assert.match(role(stale,"status").textContent,/서버 변경됨/);
            assert.equal(role(stale,"add").disabled,true);
            assert.equal(node.applied,undefined);
            role(stale,"close").click();
        """)

    def test_delete_requires_visible_target_confirmation_and_escape_backs_out(self):
        self.run_contract(r"""
            let records=[profile("alpha")],writes=0,release;
            api.fetchApi=async(path,options)=>{
                if(path.endsWith("/manage")){
                    writes++;assert.equal(JSON.parse(options.body).profile_id,"alpha");
                    await new Promise(resolve=>{release=resolve;});records=[];return response({deleted:true});
                }
                return response(path.includes("clip-loader")?clips:{profiles:records});
            };
            const dialog=await openPromptProfileManager(node,adapter);
            role(dialog,"delete").click();assert.equal(writes,0);
            assert.match(role(dialog,"delete-target").textContent,/alpha/);
            assert.equal(dialog.dispatch("cancel").defaultPrevented,true);
            assert.equal(dialog.open,true);assert.equal(role(dialog,"delete-confirm").parent.hidden,true);
            role(dialog,"delete").click();role(dialog,"delete-confirm").click();await settle();
            assert.equal(writes,1);assert.equal(role(dialog,"close").disabled,true);
            assert.equal(dialog.dispatch("cancel").defaultPrevented,true);
            release();await settle();
            assert.equal(dialog.open,false);assert.equal(node.widgets[0].value,"");
            assert.equal(writes,1);
        """)

    def test_loading_can_close_and_saved_draft_remains_readable_during_outage(self):
        self.run_contract(r"""
            api.fetchApi=async path=>response(path.includes("clip-loader")?clips:{profiles:[profile("alpha")]});
            const first=await openPromptProfileManager(node,adapter);
            role(first,"edit").click();role(first,"system-prompt").value="retain this draft";
            role(first,"close").click();
            api.fetchApi=async()=>{throw new Error("offline");};
            const offline=await openPromptProfileManager(node,adapter);
            assert.equal(role(offline,"system-prompt").value,"retain this draft");
            assert.equal(role(offline,"system-prompt").disabled,false);
            assert.equal(role(offline,"save").disabled,true);
            assert.equal(document.activeElement,role(offline,"retry"));
            role(offline,"close").click();
            const releases=[];
            api.fetchApi=()=>new Promise(resolve=>{releases.push(resolve);});
            const opening=openPromptProfileManager(node,adapter);
            const loading=node._remotePromptManagerDialog;
            assert.equal(role(loading,"close").disabled,false);
            assert.equal(loading.dispatch("cancel").defaultPrevented,false);
            loading.close();
            releases[0](response({profiles:[profile("alpha")]}));
            releases[1](response(clips));
            await opening;
            assert.equal(node.applied,undefined);
        """)

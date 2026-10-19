"""Remote model definitions must not report valid peer models as missing."""
import shutil
import unittest
from web_harness import run_web


@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class RemoteModelDefinitionsTests(unittest.TestCase):
    def test_peer_paths_and_matching_defaults_reach_both_renderers(self):
        run_web('''
          import assert from 'node:assert/strict';
          import {discoverMiniMaxModelDefinitions, applyMiniMaxModelDefinitions} from './custom/web/remote_model_definitions.js';
          function definition() {return {name:'RemoteMiniMaxH3ReferenceEncode', input:{required:{
            server_url:['STRING',{default:'http://127.0.0.1:18289'}],
            clip_name:[['h3.clip']],vae_name:[['video.vae']],audio_vae_name:[['audio.vae']]
          }}};}
          const classic=definition();
          await discoverMiniMaxModelDefinitions({RemoteMiniMaxH3ReferenceEncode:classic});
          assert.equal(classic.input.required.clip_name[1].default,'nvmes/h3.clip');
          assert.equal(classic.input.required.vae_name[1].default,'nvmes/video.vae');
          assert.equal(classic.input.required.audio_vae_name[1].default,'nvmes/audio.vae');
          const vue=definition();applyMiniMaxModelDefinitions([vue]);
          assert.deepEqual(vue.input.required.clip_name[0],['other.clip','nvmes/h3.clip']);
          vue.input.required.clip_name[1].default='other.clip';
          applyMiniMaxModelDefinitions([vue]);
          assert.equal(vue.input.required.clip_name[1].default,'other.clip');
        ''', api="""export const api={fetchApi:async path=>({ok:true,json:async()=>path.includes('clip-loader')?{clip_name:['other.clip','nvmes/h3.clip']}:{vae_name:['nvmes/audio.vae','nvmes/video.vae']}})};""")

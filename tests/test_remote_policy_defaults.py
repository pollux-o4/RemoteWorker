"""Legacy saved workflows with empty policy combos remain executable."""
import shutil
import unittest
from web_harness import run_web

@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class PolicyDefaultsTests(unittest.TestCase):
    def test_only_empty_saved_policies_are_migrated(self):
        run_web('''
          import assert from 'node:assert/strict';
          import {restorePolicyDefaults} from './custom/web/remote_policy_defaults.js';
          const widgets=[{name:'peer_memory_policy',value:''},{name:'cache_policy',value:null}];
          restorePolicyDefaults({widgets});
          assert.deepEqual(widgets.map(w=>w.value),['release_after_result','reuse_same_inputs']);
          widgets[0].value='keep';widgets[1].value='always_refresh';
          restorePolicyDefaults({widgets});
          assert.deepEqual(widgets.map(w=>w.value),['keep','always_refresh']);
          widgets[0].value='unknown';restorePolicyDefaults({widgets});
          assert.equal(widgets[0].value,'unknown');
          restorePolicyDefaults({});
        ''')

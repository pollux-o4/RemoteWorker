import shutil
import unittest

from web_harness import run_web


@unittest.skipUnless(shutil.which("node"), "Node.js required")
class SchedulerUI(unittest.TestCase):
    def test_step_boundary_and_late_clip_priority_are_explained(self):
        run_web("""
        import assert from 'node:assert/strict';
        import {schedulerText} from './custom/web/scheduler_view.js';
        assert.equal(schedulerText(null),null);
        const late=schedulerText({decision:{reason:'clip_required',remaining_steps:0}});
        assert.equal(late.title,'다음 CLIP 우선 처리');
        assert.equal(late.timing,'샘플링 0 step 남음');
        const finished=schedulerText({decision:{reason:'vae_window',prompt_id:'A',remaining_steps:2},sampling:{prompt_id:'A',step:128,total:128}});
        assert.equal(finished.timing,'샘플링 0 step 남음');
        assert.match(schedulerText({decision:{reason:'vae_ready'}}).description,/CLIP을 먼저/);
        assert.equal(schedulerText({decision:{reason:'sampling_progress'}}).timing,'');
        """)

"""Bounded local compute and remote waits through the same runtime slot interface."""

import threading
import unittest

from remoteworker.pipeline.slots import Slots


class PipelineSlots(unittest.TestCase):
    def test_next_sampling_runs_during_previous_decode_and_context_is_restored(self):
        state = {"prompt": None}
        slots = Slots(lambda job: dict(state), lambda job, saved: state.update(saved))
        decoding = threading.Event()
        sampled = threading.Event()
        results = []
        failures = []

        def first():
            try:
                with slots.lock, slots.job("A", object()):
                    state["prompt"] = "A"

                    def decode():
                        decoding.set()
                        if not sampled.wait(2):
                            raise RuntimeError("Next sampling blocked by remote decode")
                        return "decoded-A"

                    results.append(slots.remote(decode))
                    self.assertEqual(state["prompt"], "A")
            except BaseException as exc:
                failures.append(exc)

        def second():
            if not decoding.wait(2):
                failures.append(RuntimeError("Decode never started"))
                return
            with slots.lock, slots.job("B", object()):
                state["prompt"] = "B"
                self.assertEqual(len(slots.status()["jobs"]), 2)
                self.assertEqual(slots.status()["local_prompt_id"], "B")
                sampled.set()

        threads = [threading.Thread(target=first), threading.Thread(target=second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])
        self.assertEqual(results, ["decoded-A"])
        self.assertEqual(slots.status()["jobs"], [])

    def test_remote_failure_reacquires_local_slot_before_unwinding(self):
        restored = []
        slots = Slots(
            lambda job: "saved",
            lambda job, value: restored.append((job.prompt_id, value)),
        )
        with slots.lock, slots.job("A", None):
            with self.assertRaisesRegex(ValueError, "remote failed"):
                slots.remote(lambda: (_ for _ in ()).throw(ValueError("remote failed")))
            self.assertTrue(slots.lock.locked())
            self.assertEqual(slots.status()["local_prompt_id"], "A")
        self.assertEqual(restored, [("A", "saved")])


class ModelReuseTests(unittest.TestCase):
    def test_same_signature_reuses_model_but_different_lora_signature_misses(self):
        from remoteworker.pipeline.reuse import ModelReuse

        class Model:
            pass

        reuse = ModelReuse()
        model = Model()
        reuse.remember(("Loader", "lora-A"), [[model], [None]])
        self.assertIs(reuse.recall(("Loader", "lora-A"))[0][0], model)
        self.assertIsNone(reuse.recall(("Loader", "lora-B")))

    def test_weak_entries_do_not_pin_models_or_bypass_native_eviction(self):
        import gc

        from remoteworker.pipeline.reuse import ModelReuse

        class Model:
            pass

        reuse = ModelReuse()
        model = Model()
        reuse.remember("model", [[model]])
        del model
        gc.collect()
        self.assertIsNone(reuse.recall("model"))


class PipelineUX(unittest.TestCase):
    def test_pipeline_status_identifies_local_and_remote_jobs(self):
        from web_harness import run_web

        run_web("""
            import assert from "node:assert/strict";
            import {schedulerText} from "./custom/web/scheduler_view.js";
            const result=schedulerText({}, {enabled:true,local_prompt_id:"BBBBBBBB",jobs:[
              {prompt_id:"AAAAAAAA",phase:"remote"},{prompt_id:"BBBBBBBB",phase:"local"}]});
            assert.match(result.title,/병행/);
            assert.match(result.timing,/AAAAAAAA.*원격/);
            assert.match(result.timing,/BBBBBBBB.*로컬/);
            assert.match(schedulerText({}, {enabled:true,admission:"ram_pressure",jobs:[]}).title,/메모리/);
        """)

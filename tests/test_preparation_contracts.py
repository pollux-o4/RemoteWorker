"""Planning and invocation boundaries shared by background and foreground callers."""
import unittest

from remoteworker.preparation import plan
from remoteworker.preparation.invocation import (
    call_values, is_speculative, speculative_call,
)


class PreparationContracts(unittest.TestCase):
    def test_head_reorder_and_completed_attempts(self):
        node = {"class_type": "RemoteConditioning", "inputs": {"server_url": "http://peer"}}
        graph = {"1": node, "unused": dict(node)}
        a = (2, "A", graph, {}, ["1"])
        b = (1, "B", graph, {}, ["1"])
        attempts = list(plan.pending_encoders([a, b], []))
        self.assertEqual([(x[0][1], x[2]) for x in attempts], [("B", "1")])
        self.assertEqual(list(plan.pending_encoders([a, b], [attempts[0][-1]])), [])
        self.assertEqual(next(plan.pending_encoders([a], []))[0][1], "A")

    def test_nested_speculation_restores_on_failure(self):
        self.assertFalse(is_speculative())
        with speculative_call():
            with self.assertRaises(RuntimeError):
                with speculative_call():
                    raise RuntimeError("cancelled")
            self.assertTrue(is_speculative())
        self.assertFalse(is_speculative())

    def test_handoff_signature_defaults_match_actual_call(self):
        def run(self, prompt, cache_policy="reuse", *, unique_id=None):
            pass
        self.assertEqual(call_values(run, {"prompt": "hello"}), {
            "prompt": "hello", "cache_policy": "reuse", "unique_id": None,
        })

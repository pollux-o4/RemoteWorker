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

    def test_native_is_changed_annotation_does_not_cancel_promoted_head(self):
        from types import SimpleNamespace
        from remoteworker.preparation.coordinator import Coordinator
        graph={'clip':{'class_type':'RemoteConditioning','inputs':{'prompt':'B'},'is_changed':[float('nan')]}}
        copied={'clip':{'class_type':'RemoteConditioning','inputs':{'prompt':'B'}}}
        server=SimpleNamespace(prompt_queue=SimpleNamespace(get_current_queue=lambda: ([(1,'B',copied)],[])))
        owner=Coordinator(server,{},None)
        self.assertFalse(owner.preparation_cancelled('A','B',graph))
        copied['clip']['inputs']['prompt']='changed'
        self.assertTrue(owner.preparation_cancelled('A','B',graph))

    def test_ready_conditioning_is_consumed_while_peer_gate_is_busy(self):
        import threading
        from types import SimpleNamespace
        from unittest.mock import patch
        from remoteworker.preparation import integration
        from remoteworker.preparation.gates import gate
        from remoteworker.preparation.handoff import Handoffs,pack,signature
        handoffs=Handoffs();result=[];entered=threading.Event()
        owner=SimpleNamespace(handoffs=handoffs,record=lambda *a,**k:None,scheduler=None)
        def run(self,prompt,server_url,unique_id=None):
            raise AssertionError('Prepared encoder was computed again')
        managed=integration.managed(run,'encoder')
        def consume():
            entered.set()
            result.append(managed(None,'B','http://busy-handoff-peer','clip'))
        with patch.object(integration,'coordinator',owner),patch.object(integration,'current_prompt',return_value='B'):
            with gate('http://busy-handoff-peer').enter():
                thread=threading.Thread(target=consume);thread.start()
                self.assertTrue(entered.wait(1))
                handoffs.put(('B','clip'),signature({'prompt':'B','server_url':'http://busy-handoff-peer'}),pack(('ready',)))
                thread.join(1)
                ready=not thread.is_alive()
            thread.join(1)
        self.assertTrue(ready,'Encoder waited for an unrelated VAE decode despite prepared conditioning')
        self.assertEqual(result,[('ready',)])

    def test_pipeline_prefetch_does_not_consume_another_jobs_native_interrupt(self):
        from collections import deque
        from types import SimpleNamespace
        from remoteworker.core.cancellation import cancellation_scope,caller_interrupt_enabled
        from remoteworker.preparation.actions import prefetch
        observed=[]
        class Encoder:
            def run(self,server_url):
                observed.append(caller_interrupt_enabled());return ('ready',)
        owner=SimpleNamespace(done=deque(),preparation_cancelled=lambda *a:False,
            scheduler=None,record=lambda *a,**k:None,encoders={'RemoteConditioning':Encoder},handoffs=None)
        node={'class_type':'RemoteConditioning','inputs':{'server_url':'http://isolated-interrupt-peer'}}
        with cancellation_scope(lambda:False,caller_interrupt=False):
            prefetch(owner,(0,'A',{}),(1,'B',{'clip':node}),{'clip':node},'clip',node,node['inputs'],('B','clip','sig'),allow_after_sampling=True)
        self.assertEqual(observed,[False])

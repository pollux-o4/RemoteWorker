"""Queue lookahead contracts exercised through the production coordinator."""
import unittest
from types import SimpleNamespace
from remoteworker.preparation.coordinator import Coordinator
from remoteworker.preparation import plan


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        class Encoder:
            def run(self, **values):
                calls.append(('clip', values['prompt']))
        self.graph = {
            'sample': {'class_type': 'SamplerCustomAdvanced', 'inputs': {}},
            'decode': {'class_type': 'RemoteVAEDecode', 'inputs': {
                'samples': ['sample', 0], 'server_url': 'http://127.0.0.1:18289', 'vae_name': 'h3.vae'}},
        }
        self.next_graph = {'clip': {'class_type': 'RemoteConditioning', 'inputs': {
            'prompt': 'different prompt', 'server_url': 'http://127.0.0.1:18289'}}}
        self.running = [(0, 'current', self.graph)]
        self.pending = [(1, 'next', self.next_graph)]
        server = SimpleNamespace(last_node_id='sample', port=18290,
            prompt_queue=SimpleNamespace(get_current_queue=lambda: (self.running, self.pending)))
        self.coordinator = Coordinator(server, {'RemoteConditioning': Encoder},
            lambda url, model, kind: self.calls.append(('vae', model)))

    def test_next_conditioning_before_automatic_vae_without_signal(self):
        self.coordinator.tick()
        self.assertEqual(self.calls, [('clip', 'different prompt'), ('vae', 'h3.vae')])
        self.coordinator.tick()
        self.assertEqual(len(self.calls), 2)

    def test_no_next_job_still_warms_vae(self):
        self.pending.clear()
        self.coordinator.tick()
        self.assertEqual(self.calls, [('vae', 'h3.vae')])

    def test_dynamic_inputs_never_executed_speculatively(self):
        self.next_graph['clip']['inputs']['prompt'] = ['runtime_llm', 0]
        self.coordinator.tick()
        self.assertEqual(self.calls, [('vae', 'h3.vae')])
        self.assertIn('skipped', [x['state'] for x in self.coordinator.status()['events']])

    def test_no_work_outside_sampling(self):
        self.coordinator.server.last_node_id = 'decode'
        self.coordinator.tick()
        self.assertEqual(self.calls, [])

    def test_manual_ticket_takes_precedence(self):
        self.graph['decode']['inputs']['warmup'] = ['manual', 0]
        self.pending.clear()
        self.coordinator.tick()
        self.assertEqual(self.calls, [])

    def test_same_process_peer_is_not_queued(self):
        self.next_graph['clip']['inputs']['server_url'] = 'http://127.0.0.1:18290'
        self.pending.clear()
        self.graph['decode']['inputs']['server_url'] = 'http://127.0.0.1:18290'
        self.coordinator.tick()
        self.assertEqual(self.calls, [])

    def test_refresh_and_changed_prompt_are_distinct(self):
        self.next_graph['clip']['inputs']['cache_policy'] = 'always_refresh'
        self.assertIsNone(plan.literal_inputs(self.next_graph['clip']))
        before = plan.fingerprint(self.next_graph['clip'])
        self.next_graph['clip']['inputs']['prompt'] = 'new content'
        self.assertNotEqual(before, plan.fingerprint(self.next_graph['clip']))

    def test_video_vae_before_audio_even_if_graph_insertion_reversed(self):
        audio = {'a': {'class_type': 'RemoteVAEDecodeAudio', 'inputs': {
            'vae_name': 'audio.vae', 'server_url': 'http://peer'}}}
        self.assertEqual(plan.first_decoder({**audio, **self.graph})[-1], 'image')


    def test_cancelled_or_modified_queue_entry_invalidates_prefetch(self):
        expected = plan.fingerprint(self.next_graph['clip'])
        self.assertTrue(self.coordinator.job_present('next', 'clip', expected))
        self.next_graph['clip']['inputs']['prompt'] = 'changed'
        self.assertFalse(self.coordinator.job_present('next', 'clip', expected))
        self.pending.clear()
        self.assertFalse(self.coordinator.job_present('next', 'clip', expected))

    def test_idle_status_does_not_claim_model_is_still_resident(self):
        self.coordinator.record('vae_ready', model='h3.vae')
        self.running.clear()
        self.pending.clear()
        self.assertIsNone(self.coordinator.status()['active'])
        self.assertEqual(self.coordinator.status()['events'][-1]['state'], 'vae_ready')

    def test_cancellation_scope_is_thread_local_and_restored(self):
        from remoteworker.core.cancellation import cancellation_scope, check_cancelled, RequestCancelled
        with cancellation_scope(lambda: True):
            with self.assertRaises(RequestCancelled):
                check_cancelled()
        check_cancelled()


    def test_busy_peer_is_not_given_speculative_jobs(self):
        self.coordinator.available = lambda url: False
        self.coordinator.tick()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.coordinator.status()['active']['state'], 'deferred')
        self.coordinator.available = lambda url: True
        self.coordinator.tick()
        self.assertEqual(self.calls[0], ('clip', 'different prompt'))


    def test_native_progress_detects_sampling_without_browser_client(self):
        from remoteworker.preparation.execution_state import sampling_in
        registry = SimpleNamespace(prompt_id='current', nodes={
            'sample': {'state': SimpleNamespace(value='running')}},
            dynprompt=SimpleNamespace(get_node=lambda key: self.graph[key]))
        self.assertTrue(sampling_in(registry, 'current'))
        self.assertFalse(sampling_in(registry, 'another'))
        registry.nodes['sample']['state'] = SimpleNamespace(value='finished')
        self.assertFalse(sampling_in(registry, 'current'))


    def test_disconnected_nodes_are_not_prefetched(self):
        graph = {**self.next_graph,
            'unused': {'class_type': 'RemoteConditioning', 'inputs': {'prompt': 'unused'}},
            'save': {'class_type': 'SaveRemoteConditioning', 'inputs': {'conditioning': ['clip', 0]}}}
        selected = plan.reachable(graph, ['save'])
        self.assertEqual({key for key, node, values in plan.next_encoders(selected)}, {'clip'})


class PeerSerializationTests(unittest.TestCase):
    def test_foreground_overtakes_waiting_speculation(self):
        import threading
        from remoteworker.preparation.gates import PeerGate
        gate = PeerGate()
        order = []
        def foreground():
            with gate.enter():
                order.append('foreground')
        def speculative():
            with gate.enter(speculative=True):
                order.append('speculative')
        with gate.enter():
            first = threading.Thread(target=foreground)
            first.start()
            with gate.condition:
                gate.condition.wait_for(lambda: gate.foreground == 1, timeout=2)
                self.assertEqual(gate.foreground, 1)
            second = threading.Thread(target=speculative)
            second.start()
        first.join(2)
        second.join(2)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(order, ['foreground', 'speculative'])

    def test_gate_releases_after_failure(self):
        from remoteworker.preparation.gates import PeerGate
        gate = PeerGate()
        with self.assertRaises(ValueError):
            with gate.enter():
                raise ValueError('peer failed')
        with gate.enter(speculative=True):
            self.assertTrue(gate.busy)
        self.assertFalse(gate.busy)

    def test_cancelled_wait_targets_only_its_own_remote_request(self):
        from unittest.mock import patch
        from remoteworker.core.cancellation import cancellation_scope, RequestCancelled
        from remoteworker.core.execution import cancel_on_remote_failure
        from remoteworker.core.transport import HttpComfyApi
        api = HttpComfyApi('http://127.0.0.1:18289', 1)
        with patch.object(api, 'cancel_prompt') as cancel, patch.object(api, '_json') as http:
            with self.assertRaises(RequestCancelled):
                with cancellation_scope(lambda: True), cancel_on_remote_failure(api, 'own-prefetch'):
                    api.wait_for_history('own-prefetch')
            cancel.assert_called_once_with('own-prefetch')
            http.assert_not_called()

    def test_cancelled_speculation_never_submits_new_peer_work(self):
        from unittest.mock import patch
        from remoteworker.core.cancellation import cancellation_scope, RequestCancelled
        from remoteworker.core.transport import HttpComfyApi
        api = HttpComfyApi('http://127.0.0.1:18289', 1)
        with patch.object(api, '_json') as http:
            with cancellation_scope(lambda: True), self.assertRaises(RequestCancelled):
                api.submit({})
            http.assert_not_called()

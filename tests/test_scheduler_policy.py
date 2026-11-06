"""Step-boundary scheduling with mandatory missing-CLIP priority."""

import unittest
from types import SimpleNamespace
from remoteworker.preparation.policy import Admission
from remoteworker.preparation.progress import SamplingProgress


class SchedulingPolicyTests(unittest.TestCase):
    def test_three_remaining_steps_keep_clip_window(self):
        self.assertEqual(
            Admission()
            .decide(remaining_steps=3, needs_clip=False, reserved=False)
            .reason,
            "clip_window",
        )

    def test_two_remaining_steps_start_vae(self):
        self.assertEqual(
            Admission()
            .decide(remaining_steps=2, needs_clip=False, reserved=False)
            .action,
            "warmup",
        )

    def test_missing_clip_wins_at_zero_steps_even_with_prepared_vae(self):
        d = Admission().decide(remaining_steps=0, needs_clip=True, reserved=True)
        self.assertEqual(d.action, "prefetch")
        self.assertEqual(d.reason, "clip_required")

    def test_unknown_progress_never_starts_vae_early(self):
        self.assertEqual(
            Admission()
            .decide(remaining_steps=None, needs_clip=False, reserved=False)
            .action,
            "wait",
        )
        self.assertEqual(
            Admission()
            .decide(remaining_steps=None, needs_clip=True, reserved=False)
            .action,
            "prefetch",
        )

    def test_reserved_vae_is_reused_when_no_clip_is_missing(self):
        self.assertEqual(
            Admission()
            .decide(remaining_steps=1, needs_clip=False, reserved=True)
            .reason,
            "vae_ready",
        )

    def test_native_progress_counts_do_not_depend_on_gpu_timing(self):
        c = SamplingProgress()
        c.update("A", "sample", 5, 128)
        self.assertEqual(c.detail("A"), {"step": 5, "total": 128})

    def test_progress_reset_and_job_isolation(self):
        c = SamplingProgress()
        c.update("A", "sample", 5, 128)
        self.assertEqual(c.detail("B"), {})
        c.reset()
        self.assertEqual(c.detail("A"), {})

    def test_native_progress_accepts_integral_float_counts_and_ignores_placeholders(self):
        c = SamplingProgress()
        c.update("A", "sample", 0.0, 1.0)
        self.assertEqual(c.detail("A"), {})
        c.update("A", "sample", 1.0, 1.0)
        self.assertEqual(c.detail("A"), {"step": 1, "total": 1})
        c.update("B", "sample", 5.0, 128.0)
        self.assertEqual(c.detail("B"), {"step": 5, "total": 128})
        for value, total in ((0.5, 1.0), (1, 128.5), (float("nan"), 128),
                             (1, float("inf")), (None, 128), ("ready", 128),
                             (True, 128), (2**64, 2**64)):
            c.update("invalid", "sample", value, total)
            self.assertEqual(c.detail("B"), {"step": 5, "total": 128})
            self.assertEqual(c.detail("invalid"), {})

    def test_float_progress_event_preserves_native_send_and_reconciliation(self):
        from threading import Event
        from unittest.mock import patch
        from remoteworker.preparation.events import EventBridge
        from remoteworker.preparation.scheduler import Scheduler
        calls = []
        registry = SimpleNamespace(prompt_id="A", dynprompt=SimpleNamespace(
            get_node=lambda key: {"class_type": "SamplerCustomAdvanced"}))
        owner = SimpleNamespace(wake=Event(), scheduler=Scheduler())
        server = SimpleNamespace(send_sync=lambda *args: calls.append(args))
        bridge = EventBridge(server, owner)
        data = {"prompt_id": "A", "nodes": {"sample": {"state": "running", "value": 0.0, "max": 1.0}}}
        with patch("remoteworker.preparation.events._progress_registry", return_value=registry):
            server.send_sync("progress_state", data, "client-A")
            self.assertEqual(owner.scheduler.sampling.detail("A"), {})
            data["nodes"]["sample"].update(value=1.0, max=1.0)
            server.send_sync("progress_state", data, "client-A")
            self.assertEqual(owner.scheduler.sampling.detail("A"), {"step": 1, "total": 1})
            data["nodes"]["sample"].update(value=5.0, max=128.0)
            server.send_sync("progress_state", data, "client-A")
            self.assertEqual(owner.scheduler.decide("A").remaining_steps, 123)
            data["nodes"]["sample"]["value"] = 5.5
            server.send_sync("progress_state", data, "client-A")
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call[0] == "progress_state" and call[1] is data and call[2] == "client-A" for call in calls))
        self.assertEqual(owner.scheduler.sampling.detail("A"), {"step": 5, "total": 128})
        self.assertEqual(bridge.errors, 0)
        self.assertTrue(owner.wake.is_set())

    def test_event_adapter_preserves_native_routing(self):
        from threading import Event
        from remoteworker.preparation.events import EventBridge

        calls = []
        server = SimpleNamespace(send_sync=lambda *a: calls.append(a))
        owner = SimpleNamespace(wake=Event())
        EventBridge(server, owner)
        server.send_sync("status", {"n": 1}, "client-A")
        self.assertEqual(calls, [("status", {"n": 1}, "client-A")])
        self.assertTrue(owner.wake.is_set())

    def test_binary_preview_keeps_native_delivery_without_string_observer_error(self):
        from threading import Event
        from remoteworker.preparation.events import EventBridge

        calls = []
        server = SimpleNamespace(send_sync=lambda *args: calls.append(args))
        owner = SimpleNamespace(wake=Event())
        bridge = EventBridge(server, owner)
        data = (1, b"preview bytes")
        server.send_sync(4, data, "client-A")
        self.assertEqual(calls, [(4, data, "client-A")])
        self.assertIs(calls[0][1], data)
        self.assertEqual(bridge.errors, 0)
        self.assertFalse(owner.wake.is_set())

    def make_coordinator(self, steps):
        from remoteworker.preparation.coordinator import Coordinator
        from remoteworker.preparation.scheduler import Scheduler

        self.calls = []
        calls = self.calls

        class Encoder:
            def run(self, **values):
                calls.append("clip")
                return ()

        self.graph = {
            "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
            "decode": {
                "class_type": "RemoteVAEDecode",
                "inputs": {
                    "samples": ["sample", 0],
                    "server_url": "http://step-peer",
                    "vae_name": "video.vae",
                },
            },
        }
        self.following = {
            "clip": {
                "class_type": "RemoteConditioning",
                "inputs": {"prompt": "B", "server_url": "http://step-peer"},
            }
        }
        self.running = [(0, "A", self.graph)]
        self.pending = []
        server = SimpleNamespace(
            port=8188,
            last_node_id="sample",
            prompt_queue=SimpleNamespace(
                get_current_queue=lambda: (self.running, self.pending)
            ),
        )
        scheduler = Scheduler()
        scheduler.sampling.update("A", "sample", 128 - steps, 128)
        c = Coordinator(
            server,
            {"RemoteConditioning": Encoder},
            lambda *_: calls.append("vae"),
            scheduler=scheduler,
        )
        return c

    def test_coordinator_waits_at_three_then_warms_at_two(self):
        c = self.make_coordinator(3)
        c.tick()
        self.assertEqual(self.calls, [])
        c.scheduler.sampling.update("A", "sample", 126, 128)
        c.tick()
        c.tick()
        self.assertEqual(self.calls, ["vae"])

    def test_late_clip_evicts_reservation_then_rewarms_once(self):
        c = self.make_coordinator(2)
        c.tick()
        self.pending.append((1, "B", self.following))
        c.tick()
        c.tick()
        self.assertEqual(self.calls, ["vae", "clip", "vae"])

    def test_inflight_clip_survives_sampling_end_but_not_queue_cancel(self):
        c = self.make_coordinator(3)
        self.pending.append((1, "B", self.following))
        c.server.last_node_id = "decode"
        self.assertFalse(c.preparation_cancelled("A", "B", self.following))
        self.pending.clear()
        self.assertTrue(c.preparation_cancelled("A", "B", self.following))

    def test_decode_barrier_handles_late_clip_before_vae(self):
        c = self.make_coordinator(0)
        self.pending.append((1, "B", self.following))
        c.server.last_node_id = "decode"
        c.ensure_next_before_decode("http://step-peer")
        self.calls.append("actual_vae")
        self.assertEqual(self.calls, ["clip", "actual_vae"])

    def test_reordered_queue_cancels_old_head_only(self):
        c = self.make_coordinator(3)
        self.pending.extend([(1, "B", self.following), (0, "C", self.following)])
        self.assertTrue(c.preparation_cancelled("A", "B", self.following))
        self.assertFalse(c.preparation_cancelled("A", "C", self.following))

    def test_new_missing_clip_can_interrupt_dummy_warmup(self):
        c = self.make_coordinator(2)
        self.pending.append((1, "B", self.following))
        self.assertTrue(c.next_clip_missing("A", "http://step-peer"))
        token = (
            "B",
            "clip",
            __import__(
                "remoteworker.preparation.plan", fromlist=["fingerprint"]
            ).fingerprint(self.following["clip"]),
        )
        c.done.append(token)
        self.assertFalse(c.next_clip_missing("A", "http://step-peer"))

    def test_handoff_budget_rejects_before_serialization(self):
        from unittest.mock import patch
        import torch
        from remoteworker.preparation.handoff import pack

        with patch.object(
            torch, "save", side_effect=AssertionError("oversized tensor serialized")
        ):
            self.assertIsNone(pack((torch.zeros(1024),), max_bytes=64))
        self.assertIsInstance(pack((torch.zeros(4),), max_bytes=4096), bytes)

    def test_cancelling_next_clip_at_decode_barrier_does_not_fail_current(self):
        from remoteworker.core.cancellation import RequestCancelled

        c = self.make_coordinator(0)
        self.pending.append((1, "B", self.following))
        c.server.last_node_id = "decode"

        def removed(*a, **kw):
            self.pending.clear()
            raise RequestCancelled()

        c.prepare_next = removed
        c.ensure_next_before_decode("http://step-peer")
        self.assertEqual(self.running[0][1], "A")

    def test_failed_next_clip_at_decode_barrier_preserves_current_decode(self):
        c = self.make_coordinator(0)
        self.pending.append((1, "B", self.following))
        c.server.last_node_id = "decode"

        def failed(*a, **kw):
            raise RuntimeError("next encoder failed")

        c.prepare_next = failed
        c.ensure_next_before_decode("http://step-peer")
        self.assertEqual(c.events[-1]["state"], "error")

    def test_single_step_sampler_still_has_boundary_progress(self):
        c = SamplingProgress()
        c.update("A", "sample", 1, 1)
        self.assertEqual(c.detail("A"), {"step": 1, "total": 1})

    def test_gate_wait_is_cancellable_without_releasing_another_owner(self):
        import threading
        from remoteworker.preparation.gates import PeerGate

        g = PeerGate()
        waiting = threading.Event()
        cancel = threading.Event()
        errors = []

        def check():
            waiting.set()
            if cancel.is_set():
                raise ValueError("cancelled")

        def waiter():
            try:
                with g.enter(check_wait=check):
                    raise AssertionError("entered busy gate")
            except ValueError:
                errors.append("cancelled")

        with g.enter():
            worker = threading.Thread(target=waiter)
            worker.start()
            self.assertTrue(waiting.wait(2))
            cancel.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertTrue(g.busy)
            self.assertEqual(g.foreground, 0)
        self.assertEqual(errors, ["cancelled"])

    def test_background_scope_does_not_consume_foreground_interrupt(self):
        import sys
        from unittest.mock import patch, Mock
        from remoteworker.core.cancellation import (
            cancellation_scope,
            caller_interrupt_enabled,
        )
        from remoteworker.core.transport import _raise_if_caller_interrupted

        model = SimpleNamespace(throw_exception_if_processing_interrupted=Mock())
        with patch.dict(
            sys.modules, {"comfy": SimpleNamespace(model_management=model)}
        ):
            with cancellation_scope(lambda: False, caller_interrupt=False):
                _raise_if_caller_interrupted()
                self.assertFalse(caller_interrupt_enabled())
            model.throw_exception_if_processing_interrupted.assert_not_called()
            _raise_if_caller_interrupted()
            model.throw_exception_if_processing_interrupted.assert_called_once()

    def test_decode_barrier_rechecks_head_after_waiting(self):
        c = self.make_coordinator(0)
        self.pending.append((1, "B", self.following))
        seen = []
        original = c.prepare_next

        def changed(*a, **kw):
            seen.append(self.pending[0][1])
            if len(seen) == 1:
                self.pending[0] = (0, "C", self.following)
            else:
                original(*a, **kw)

        c.prepare_next = changed
        c.ensure_next_before_decode("http://step-peer")
        self.assertEqual(seen, ["B", "C"])
        self.assertEqual(self.calls, ["clip"])

    def test_initial_zero_of_one_placeholder_is_not_a_remaining_step(self):
        c = SamplingProgress()
        c.update("A", "sample", 0, 1)
        self.assertEqual(c.detail("A"), {})
        c.update("A", "sample", 0, 128)
        self.assertEqual(c.detail("A"), {"step": 0, "total": 128})

    def test_cancelled_head_can_be_prepared_again_when_it_returns(self):
        from remoteworker.core.cancellation import RequestCancelled

        c = self.make_coordinator(3)
        self.pending.append((1, "B", self.following))

        class Cancelled:
            def run(self, **kw):
                raise RequestCancelled()

        original = c.encoders["RemoteConditioning"]
        c.encoders["RemoteConditioning"] = Cancelled
        with self.assertRaises(RequestCancelled):
            c.prepare_next(self.running[0])
        self.assertFalse(c.done)
        c.encoders["RemoteConditioning"] = original
        c.prepare_next(self.running[0])
        self.assertEqual(self.calls, ["clip"])

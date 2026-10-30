"""Native ownership under interleaved host callbacks and cancellation."""
import gc
import threading
import unittest

from remoteworker._native import Job, SlotState, PeerGate
from remoteworker.pipeline.slots import Slots


class NativeExecutionTests(unittest.TestCase):
    def test_slot_transitions_reject_resume_before_completion(self):
        state = SlotState()
        job = Job("A", None)
        state.start(job)
        with self.assertRaisesRegex(ValueError, "ready"):
            state.resume(job)
        state.suspend(job)
        with self.assertRaisesRegex(ValueError, "local"):
            state.suspend(job)
        state.completed(job)
        state.resume(job)
        self.assertEqual(job.phase, "local")
        self.assertEqual(state.ready, 0)
        state.finish(job)
        self.assertEqual(state.status()["jobs"], [])

    def test_restore_failure_does_not_leave_a_ready_reservation(self):
        slots = Slots(lambda job: None, lambda *_: (_ for _ in ()).throw(ValueError("restore")))
        with slots.lock:
            with self.assertRaisesRegex(ValueError, "restore"):
                with slots.job("A", None):
                    slots.remote(lambda: None)
            self.assertTrue(slots.lock.locked())
        self.assertEqual(slots.ready, 0)
        self.assertEqual(slots.status()["jobs"], [])

    def test_foreground_waiter_precedes_speculation_and_wait_releases_gil(self):
        gate = PeerGate()
        queued = threading.Event()
        speculative_queued = threading.Event()
        foreground_entered = threading.Event()
        release_foreground = threading.Event()
        order, errors = [], []

        def foreground():
            try:
                with gate.enter(check_wait=queued.set):
                    order.append("foreground")
                    foreground_entered.set()
                    if not release_foreground.wait(2):
                        raise RuntimeError("foreground release timeout")
            except BaseException as error:
                errors.append(error)

        def speculative():
            try:
                with gate.enter(speculative=True, check_wait=speculative_queued.set):
                    order.append("speculative")
            except BaseException as error:
                errors.append(error)

        with gate.enter():
            first = threading.Thread(target=foreground)
            second = threading.Thread(target=speculative)
            first.start()
            self.assertTrue(queued.wait(2))
            second.start()
            self.assertTrue(speculative_queued.wait(2))
            self.assertEqual(gate.foreground, 1)
        self.assertTrue(foreground_entered.wait(2))
        self.assertEqual(order, ["foreground"])
        release_foreground.set()
        for thread in (first, second):
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(order, ["foreground", "speculative"])
        self.assertFalse(gate.busy)
        self.assertEqual(gate.foreground, 0)

    def test_abandoned_lease_releases_owned_gate(self):
        gate = PeerGate()
        lease = gate.enter()
        lease.__enter__()
        self.assertTrue(gate.busy)
        del lease
        gc.collect()
        self.assertFalse(gate.busy)

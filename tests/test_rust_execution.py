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
            with slots.job("A", None):
                with self.assertRaisesRegex(ValueError, "restore"):
                    slots.remote(lambda: None)
                self.assertTrue(slots.lock.locked())
                self.assertEqual(slots.ready, 0)
                self.assertIsNone(slots.active)
                self.assertEqual(slots.status()["jobs"][0]["phase"], "failed")
            with slots.job("B", None):
                self.assertEqual(slots.active, "B")
        self.assertEqual(slots.ready, 0)
        self.assertEqual(slots.status()["jobs"], [])
        self.assertIsNone(slots.active)

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


class NativeHostTests(unittest.TestCase):
    def fixture(self, prompt_id="A"):
        from types import ModuleType, SimpleNamespace
        import logging
        from remoteworker.pipeline.host import ComfyHost
        host = ComfyHost(SimpleNamespace(nodes=SimpleNamespace(interrupt_processing=lambda: calls.append("interrupt"))))
        calls = []
        host.slots = Slots(lambda job: None, lambda *_: None)
        host.server = SimpleNamespace(client_id="client-A", workflow_metadata={}, send_sync=lambda *args: calls.append(("event", args)))
        host.assets = SimpleNamespace(pause_background_scan=lambda: calls.append("pause"))
        host.housekeeping = lambda: calls.append("housekeeping")
        executor = SimpleNamespace(status_messages=[], success=True, history_result={"outputs": {}})
        def execute(*args):
            calls.append(("execute", args))
            executor.success = True
            executor.history_result = {"outputs": {"save": {"images": []}}}
        executor.execute = execute
        executor.add_message = lambda *args, **kwargs: calls.append(("message", args, kwargs))
        host.executor = lambda: executor
        prompt = (0, prompt_id, {"save": {"class_type": "SaveImage"}}, {"client_id": "client-A"}, ["save"], {"auth": "secret"}, "metadata")
        items = [(prompt, "item-A")]
        def take(**kwargs):
            self.assertEqual(kwargs, {"timeout": 0.1})
            return items.pop(0) if items else None
        def done(item_id, result, **kwargs):
            calls.append(("done", item_id, result, kwargs["status"], kwargs["process_item"](prompt)))
            host.queue.currently_running.pop(item_id, None)
        host.queue = SimpleNamespace(mutex=threading.Lock(), currently_running={"item-A": prompt}, get=take, task_done=done)
        source = ModuleType("host_fixture")
        source.time = SimpleNamespace(monotonic=lambda: 100.0, perf_counter=lambda: 100.0)
        source.snapshot = lambda: {"available": 8 * 1024**3}
        source.logging = logging
        return host, executor, source, SimpleNamespace(ExecutionStatus=lambda **kwargs: SimpleNamespace(**kwargs)), calls

    def test_native_worker_executes_and_strips_sensitive_history(self):
        from remoteworker._native import host_worker_tick
        host, executor, source, queue_type, calls = self.fixture()
        self.assertIs(host_worker_tick(host, None, source, queue_type), executor)
        execution = next(value for value in calls if isinstance(value, tuple) and value[0] == "execute")
        self.assertEqual(execution[1][1], "A")
        self.assertEqual(execution[1][2]["auth"], "secret")
        completion = next(value for value in calls if isinstance(value, tuple) and value[0] == "done")
        self.assertTrue(completion[3].completed)
        self.assertEqual(len(completion[4]), 6)
        self.assertNotIn({"auth": "secret"}, completion[4])
        self.assertEqual(host.slots.status()["jobs"], [])
        self.assertIsNone(host.slots.active)
        self.assertEqual(host.server.workflow_metadata, {})

    def test_cancellation_before_slot_creation_skips_only_that_executor(self):
        from remoteworker._native import host_worker_tick
        host, executor, source, queue_type, calls = self.fixture()
        self.assertTrue(host.cancel("A"))
        self.assertTrue(host.policy.pending("A"))
        host_worker_tick(host, None, source, queue_type)
        self.assertFalse(host.policy.pending("A"))
        self.assertFalse(any(isinstance(value, tuple) and value[0] == "execute" for value in calls))
        self.assertFalse(executor.success)
        self.assertFalse(host.cancel("A"))
        self.assertFalse(host.policy.pending("A"))
        self.assertNotIn("interrupt", calls)
        self.assertEqual(host.slots.ready, 0)

    def test_ram_pressure_leaves_next_prompt_pending(self):
        from remoteworker._native import host_worker_tick
        host, executor, source, queue_type, calls = self.fixture()
        source.snapshot = lambda: {"available": 1}
        with host.slots.job("previous", None) as previous:
            host.slots.state.suspend(previous)
            host.queue.get = lambda **_: self.fail("RAM admission removed a pending prompt")
            self.assertIs(host_worker_tick(host, executor, source, queue_type), executor)
            self.assertEqual(host.admission, "ram_pressure")
            self.assertFalse(any(isinstance(value, tuple) and value[0] == "execute" for value in calls))
        self.assertEqual(host.slots.status()["jobs"], [])

    def test_execution_failure_commits_error_and_releases_local_slot(self):
        from remoteworker._native import host_worker_tick
        host, executor, source, queue_type, calls = self.fixture()
        executor.execute = lambda *_: (_ for _ in ()).throw(ValueError("model failed"))
        with self.assertLogs("host_fixture", level="ERROR"):
            host_worker_tick(host, executor, source, queue_type)
        completion = next(value for value in calls if isinstance(value, tuple) and value[0] == "done")
        self.assertEqual(completion[3].status_str, "error")
        self.assertFalse(completion[3].completed)
        self.assertEqual(host.slots.status()["jobs"], [])
        self.assertIsNone(host.slots.active)
        self.assertEqual(host.slots.ready, 0)

    def test_malformed_interrupt_never_becomes_cancel_all(self):
        from remoteworker._native import host_interrupt
        host, _, _, _, calls = self.fixture()
        for payload in ([], "A", {"prompt_id": False}, {"prompt_id": ""}, {"prompt_id": 1}):
            with self.assertRaises(ValueError):
                host_interrupt(host, payload)
        self.assertFalse(host.policy.pending("A"))
        self.assertEqual(calls, [])


class NativeShutdownTests(unittest.TestCase):
    def test_torch_host_daemon_workers_join_before_interpreter_finalization(self):
        import os
        import subprocess
        import sys
        import textwrap
        from pathlib import Path
        from remoteworker import _native
        code = textwrap.dedent('''
            import atexit, importlib.util, os, sys, threading
            from types import SimpleNamespace
            import torch
            import remoteworker
            spec = importlib.util.spec_from_file_location("remoteworker._native", os.environ["TEST_NATIVE_BINARY"])
            native = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = native
            spec.loader.exec_module(native)
            remoteworker._native = native
            from remoteworker.pipeline.host import ComfyHost
            sys.modules["execution"] = SimpleNamespace(PromptQueue=SimpleNamespace())
            host = ComfyHost(SimpleNamespace())
            host.executor = lambda: torch.zeros(2, 2)
            host.housekeeping = lambda: None
            ready = threading.Event()
            def take(**kwargs):
                ready.set()
                threading.Event().wait(0.1)
                return None
            queue = SimpleNamespace(get=take)
            worker = threading.Thread(target=host.run, args=(queue, None, None), daemon=True)
            def check_shutdown():
                if worker.is_alive() or any(t.is_alive() for t in host._threads) or not host.policy.stopping:
                    os._exit(72)
                print("joined native workers before finalization", flush=True)
            atexit.register(check_shutdown)
            worker.start()
            assert ready.wait(3)
        ''')
        result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20,
            env={**os.environ, "REMOTEWORKER_CONFIG": "", "TEST_NATIVE_BINARY": _native.__file__})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("joined native workers before finalization", result.stdout)
        self.assertNotIn("Fatal Python error", result.stderr)

    def test_shutdown_cancels_running_job_and_blocks_new_admission(self):
        from remoteworker._native import host_worker_tick
        host, executor, source, queue_type, calls = NativeHostTests().fixture()
        with host.slots.lock, host.slots.job("A", executor) as job:
            host.shutdown()
            self.assertTrue(host.policy.stopping)
            self.assertTrue(job.cancel.is_set())
            self.assertEqual(calls, ["interrupt"])
        host.queue.get = lambda **kwargs: self.fail("shutdown admitted pending work")
        self.assertIs(host_worker_tick(host, executor, source, queue_type), executor)
        self.assertEqual(host.slots.status()["jobs"], [])

    def test_coordinator_shutdown_wakes_and_joins_without_waiting_reconcile_interval(self):
        from types import SimpleNamespace
        from remoteworker.preparation.coordinator import Coordinator
        owner = Coordinator(SimpleNamespace(), {}, None)
        owner.tick = lambda: None
        owner.start()
        owner.shutdown()
        self.assertTrue(owner.stop.is_set())
        self.assertFalse(owner.thread.is_alive())
        self.assertTrue(owner.preparation_cancelled("A", "B", {}))

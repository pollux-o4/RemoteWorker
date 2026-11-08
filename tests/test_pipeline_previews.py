"""An A preview must be visible while B still owns the local GPU slot."""

import threading
import unittest
from types import SimpleNamespace

from remoteworker.pipeline.previews import Previews
from remoteworker.pipeline.slots import Slots


class PipelinePreviews(unittest.TestCase):
    def test_preview_arrives_before_next_sampler_finishes_and_encodes_once(self):
        sent, encoded = [], []

        class Preview:
            def save_images(self, images, **kwargs):
                encoded.append(kwargs)
                return {"ui": {"images": ["temp.png"]}, "result": (images,)}

        state = {"prompt": "A"}
        slots = Slots(lambda job: dict(state), lambda job, saved: state.update(saved))
        context = SimpleNamespace(node_id="p", prompt_id="A")
        previews = Previews(
            Preview,
            {"PreviewImage": Preview},
            lambda *args: sent.append(args),
            lambda: getattr(slots.thread, "job", None),
            lambda: context,
        )
        previews.install()
        images = SimpleNamespace(device=SimpleNamespace(type="cpu"), ndim=4)
        graph = {
            "d": {"class_type": "Decode", "inputs": {}},
            "p": {"class_type": "PreviewImage", "inputs": {"images": ["d", 0]}},
            "disabled": {"class_type": "PreviewImage", "inputs": {"images": ["d", 0]}},
        }
        dyn = SimpleNamespace(
            original_prompt=graph,
            all_node_ids=lambda: graph.keys(),
            get_node=graph.__getitem__,
            get_display_node_id=lambda key: "display-" + key,
        )
        decoding, b_started, preview_sent, finish_b = (threading.Event() for _ in range(4))
        failures = []

        def first():
            try:
                with slots.lock, slots.job("A", None) as job:

                    def decode():
                        decoding.set()
                        self.assertTrue(b_started.wait(2))
                        return (images, None)

                    def publish(value, saved):
                        previews.publish(job, dyn, "d", value, {"client_id": "client-A"}, ["p"])
                        preview_sent.set()

                    slots.remote(decode, completed=publish)
                    self.assertEqual(state["prompt"], "A")
                    reused = Preview().save_images(images, prompt=graph)
                    self.assertEqual(reused["ui"], {"images": ["temp.png"]})
            except BaseException as exc:
                failures.append(exc)

        def second():
            if decoding.wait(2):
                with slots.lock, slots.job("B", None):
                    state["prompt"] = "B"
                    b_started.set()
                    finish_b.wait(3)

        a = threading.Thread(target=first)
        a.start()
        # A yields its lock before B starts.
        b = threading.Thread(target=second)
        b.start()
        try:
            self.assertTrue(preview_sent.wait(2), "preview blocked behind B sampling")
            self.assertFalse(finish_b.is_set())
            self.assertEqual(state["prompt"], "B", "publishing overwrote B execution state")
            self.assertEqual(sent[0][0], "executed")
            self.assertEqual(sent[0][1]["prompt_id"], "A")
            self.assertEqual(sent[0][1]["display_node"], "display-p")
            self.assertEqual(sent[0][2], "client-A")
            self.assertEqual(len(sent), 1, "disabled output was previewed")
            self.assertTrue(any(j.get("preview_nodes") == ["p"] for j in slots.status()["jobs"]))
        finally:
            finish_b.set()
            a.join(3)
            b.join(3)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(encoded), 1)

    def test_cancelled_transformed_and_final_saving_nodes_are_not_run_early(self):
        calls = []

        class Preview:
            def save_images(self, images, **kwargs):
                calls.append("preview")
                return {"ui": {"images": ["temp.png"]}, "result": (images,)}

        class Save(Preview):
            pass

        slots = Slots(lambda job: None, lambda job, saved: None)
        previews = Previews(
            Preview,
            {"PreviewImage": Preview, "SaveImage": Save},
            lambda *args: calls.append("event"),
            lambda: None,
            lambda: None,
        )
        graph = {
            "d": {"class_type": "Decode", "inputs": {}},
            "t": {"class_type": "Transform", "inputs": {"images": ["d", 0]}},
            "p": {"class_type": "PreviewImage", "inputs": {"images": ["t", 0]}},
            "s": {"class_type": "SaveImage", "inputs": {"images": ["d", 0]}},
        }
        dyn = SimpleNamespace(
            original_prompt=graph,
            all_node_ids=lambda: graph.keys(),
            get_node=graph.__getitem__,
            get_display_node_id=lambda key: key,
        )
        images = SimpleNamespace(device=SimpleNamespace(type="cpu"), ndim=4)
        with slots.lock, slots.job("A", None) as job:
            previews.publish(job, dyn, "d", (images,), {"client_id": "client-A"}, ["p", "s"])
            self.assertEqual(calls, [])
            graph["p"]["inputs"]["images"] = ["d", 0]
            job.cancel.set()
            previews.publish(job, dyn, "d", (images,), {"client_id": "client-A"}, ["p"])
            self.assertEqual(calls, [])

    def test_preview_failure_defers_to_native_node_and_does_not_fail_decode(self):
        class Preview:
            def save_images(self, images, **kwargs):
                raise OSError("disk full")

        slots = Slots(lambda job: None, lambda job, saved: None)
        previews = Previews(
            Preview,
            {"PreviewImage": Preview},
            lambda *args: None,
            lambda: None,
            lambda: None,
        )
        graph = {
            "d": {"class_type": "Decode", "inputs": {}},
            "p": {"class_type": "PreviewImage", "inputs": {"images": ["d", 0]}},
        }
        dyn = SimpleNamespace(
            original_prompt=graph,
            all_node_ids=lambda: graph.keys(),
            get_node=graph.__getitem__,
        )
        images = SimpleNamespace(device=SimpleNamespace(type="cpu"), ndim=4)
        with (
            slots.lock,
            slots.job("A", None) as job,
            self.assertLogs("remoteworker.pipeline.previews", level="ERROR"),
        ):
            previews.publish(job, dyn, "d", {"result": (images,)}, {"client_id": "client-A"}, ["p"])
            self.assertEqual(job.previews, {})

    def test_api_without_client_id_never_broadcasts_preview(self):
        def unexpected(*args, **kwargs):
            self.fail("preview leaked to other websocket clients")

        previews = Previews(
            SimpleNamespace(save_images=unexpected),
            {},
            unexpected,
            lambda: None,
            lambda: None,
        )
        # No graph/tensor access or rendering is needed without a recipient.
        previews.publish(None, None, "d", None, {}, [])

    def test_native_list_mapping_reuses_each_preview_and_accumulates_early_images(self):
        renders, events = [], []

        class Preview:
            def save_images(self, images, **kwargs):
                name = f"{len(renders)}.png"
                renders.append(name)
                return {"ui": {"images": [name]}, "result": (images,)}

        slots = Slots(lambda job: None, lambda job, saved: None)
        context = SimpleNamespace(node_id="p", prompt_id="A", list_index=0)
        previews = Previews(
            Preview,
            {"PreviewImage": Preview},
            lambda *args: events.append(args),
            lambda: getattr(slots.thread, "job", None),
            lambda: context,
        )
        previews.install()
        images = SimpleNamespace(device=SimpleNamespace(type="cpu"), ndim=4)
        graph = {
            "d": {"class_type": "Decode", "inputs": {}},
            "p": {"class_type": "PreviewImage", "inputs": {"images": ["d", 0]}},
        }
        dyn = SimpleNamespace(
            original_prompt=graph,
            all_node_ids=lambda: graph.keys(),
            get_node=graph.__getitem__,
            get_display_node_id=lambda key: key,
        )
        with slots.lock, slots.job("A", None) as job:
            for index in range(2):
                previews.publish(
                    job,
                    dyn,
                    "d",
                    (images,),
                    {"client_id": "client-A"},
                    ["p"],
                    list_index=index,
                )
            self.assertEqual(events[-1][1]["output"]["images"], ["0.png", "1.png"])
            for index in range(2):
                context.list_index = index
                self.assertEqual(Preview().save_images(images)["ui"]["images"], [f"{index}.png"])
            self.assertEqual(job.preview_nodes, ["p"])
        self.assertEqual(len(renders), 2)

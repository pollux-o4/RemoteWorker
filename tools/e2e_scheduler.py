"""Real GPU acceptance for CLIP-first scheduling and the two-step VAE boundary."""

import json
import sys
import time

from e2e_i2v_preparation import (
    CALLER,
    PEER,
    REPORTS,
    ROOT,
    create_fixtures,
    graph,
    request,
    validate_short_video,
    wait_jobs,
)


def status():
    return request(CALLER, "/remoteworker/preparation")


def submit(label, steps=1, front=False):
    return request(CALLER, "/prompt", {"prompt": graph("steps-" + label, steps), "front": front})["prompt_id"]


def wait_until(check, seconds=600):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        try:
            value = check()
        except OSError:
            value = None
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError("step scheduler condition timed out")


def complete(ids, name):
    d = wait_jobs(ids, "step-policy-" + name + ".json")
    assert all(j["status"]["status_str"] == "success" for j in d["jobs"].values()), {p: j["status"]["status_str"] for p, j in d["jobs"].items()}
    for j in d["jobs"].values():
        validate_short_video(j, ROOT / "output")
    assert d["preparation"]["handoff_bytes"] == 0
    return d


def events(d, p, state):
    return [e for e in d["preparation"]["events"] if e.get("prompt_id") == p and e["state"] == state]


def assert_handoff(d, b, *, allow_cancel=False):
    ready = events(d, b, "conditioning_ready")
    used = events(d, b, "conditioning_consumed")
    assert ready and len(used) == 1
    start = next(v["timestamp"] / 1000 for k, v in d["jobs"][b]["status"]["messages"] if k == "execution_start")
    assert ready[0]["at"] < start <= used[0]["at"]
    if not allow_cancel:
        assert not events(d, b, "cancelled")


def near_end(a):
    def check():
        p = status()["scheduler"]["sampling"]
        return p if p.get("prompt_id") == a and p.get("total", 0) > 1 and 0 < p["total"] - p["step"] <= 2 else None

    return wait_until(check)


def conditioning_started(b):
    return wait_until(lambda: any(e["state"] == "conditioning" and e.get("prompt_id") == b for e in status()["events"]))


def boundary():
    a = submit("boundary", 64)
    print("boundary", a, flush=True)
    d = complete([a], "boundary")
    warm = events(d, a, "vae")
    assert warm and all(e.get("remaining_steps") is not None and e["remaining_steps"] <= 2 and e.get("total_steps") == 64 for e in warm), warm
    print("PASS VAE starts only at <=2 remaining steps", warm, flush=True)


def early():
    a = submit("early", 64)
    time.sleep(3)
    b = submit("early-next")
    print("early +3s", a, b, flush=True)
    d = complete([a, b], "early")
    assert_handoff(d, b)
    print("PASS unknown timings do not block CLIP; one-shot handoff", flush=True)


def late():
    a = submit("late", 128)
    p = near_end(a)
    b = submit("late-next")
    print("late <=2", p, a, b, flush=True)
    conditioning_started(b)

    def sampling_done_clip_active():
        s = status()
        p = s["scheduler"]["sampling"]
        active = s.get("active") or {}
        return (
            s
            if p.get("prompt_id") == a and p.get("step") == p.get("total") and active.get("prompt_id") == b and active.get("state") == "conditioning"
            else None
        )

    waiting = wait_until(sampling_done_clip_active)
    (REPORTS / "step-policy-late-barrier.json").write_text(json.dumps(waiting, indent=2))
    d = complete([a, b], "late")
    assert_handoff(d, b)
    print(
        "PASS late missing CLIP survives sampling end; VAE waits; A and B success",
        flush=True,
    )


def cancel():
    a = submit("cancel", 128)
    near_end(a)
    b = submit("cancel-next")
    conditioning_started(b)
    request(CALLER, "/queue", {"delete": [b]})
    d = complete([a], "cancel")
    assert events(d, b, "cancelled")
    c = submit("replay")
    complete([c], "replay")
    print(
        "PASS deleting late B preserves A decode; different C replay success",
        flush=True,
    )


def current_cancel():
    a = submit("current-cancel", 128)
    near_end(a)
    b = submit("current-cancel-next")
    conditioning_started(b)
    wait_until(lambda: (p := status()["scheduler"]["sampling"]).get("prompt_id") == a and p.get("step") == p.get("total"))
    request(CALLER, "/interrupt", {"prompt_id": a})
    ah = wait_until(lambda: request(CALLER, "/history/" + a).get(a))
    assert any(k == "execution_interrupted" for k, _ in ah["status"]["messages"]), ah["status"]
    complete([b], "current-cancel")
    assert not request(CALLER, "/queue")["queue_pending"]
    (REPORTS / "step-policy-current-cancelled.json").write_text(json.dumps({"prompt_id": a, "status": ah["status"]}, indent=2))
    print("PASS A cancelled while decode waits for CLIP; valid B completes", flush=True)


def reorder():
    a = submit("reorder", 128)
    near_end(a)
    b = submit("reorder-b")
    conditioning_started(b)
    c = submit("reorder-c", front=True)
    print("front insertion", a, b, c, flush=True)
    d = complete([a, c, b], "reorder")
    assert events(d, b, "cancelled")
    assert_handoff(d, c)
    assert_handoff(d, b, allow_cancel=True)
    cs = next(v["timestamp"] for k, v in d["jobs"][c]["status"]["messages"] if k == "execution_start")
    bs = next(v["timestamp"] for k, v in d["jobs"][b]["status"]["messages"] if k == "execution_start")
    assert cs < bs
    print(
        "PASS new head C replaces B preparation; A/C/B success in queue order",
        flush=True,
    )


if __name__ == "__main__":
    create_fixtures()
    for url in (CALLER, PEER):
        wait_until(lambda url=url: request(url, "/system_stats"))
    for name in sys.argv[1:] or [
        "boundary",
        "early",
        "late",
        "cancel",
        "current_cancel",
        "reorder",
    ]:
        globals()[name]()

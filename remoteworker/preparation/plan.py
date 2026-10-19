"""Pure queue planning; never evaluate arbitrary graph nodes speculatively."""

import hashlib
import json

ENCODERS = frozenset(
    {
        "RemoteConditioning",
        "RemoteMiniMaxH3ReferenceEncode",
        "RemoteMiniMaxH3ImageToVideo",
    }
)
DECODERS = {"RemoteVAEDecode": "image", "RemoteVAEDecodeAudio": "audio"}


def literal_inputs(node):
    values = node.get("inputs", {})
    # Links can contain images, runtime text, model overrides or other side effects.
    if any(isinstance(v, (list, dict)) for v in values.values()):
        return None
    if values.get("cache_policy", "reuse") == "always_refresh":
        return None
    return dict(values)


def fingerprint(node):
    values = {
        k: v for k, v in node.get("inputs", {}).items() if k not in ("timeout_seconds",)
    }
    return hashlib.sha256(
        json.dumps([node["class_type"], values], sort_keys=True).encode()
    ).hexdigest()


def is_sampling(graph, node_id):
    name = graph.get(str(node_id), {}).get("class_type", "")
    return "Sampler" in name or name == "SamplerCustom"


def next_encoders(graph):
    return [
        (key, node, literal_inputs(node))
        for key, node in graph.items()
        if node.get("class_type") in ENCODERS
    ]


def first_decoder(graph):
    # Warm only the first decoder: loading both H3 VAEs can evict the first on 8 GB.
    candidates = [
        (key, node) for key, node in graph.items() if node.get("class_type") in DECODERS
    ]
    candidates.sort(key=lambda item: DECODERS[item[1]["class_type"]] != "image")
    for key, node in candidates:
        values = node.get("inputs", {})
        if values.get("warmup") is not None:  # Explicit manual ordering wins.
            continue
        if all(
            isinstance(values.get(k), str) and values[k]
            for k in ("vae_name", "server_url")
        ):
            return (
                key,
                values["server_url"],
                values["vae_name"],
                DECODERS[node["class_type"]],
            )
    return None


def reachable(graph, outputs=None):
    if outputs is None:
        return graph
    visited = set()
    todo = list(outputs)
    while todo:
        key = str(todo.pop())
        if key in visited or key not in graph:
            continue
        visited.add(key)
        for value in graph[key].get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                todo.append(value[0])
    return {key: node for key, node in graph.items() if key in visited}


def head(pending):
    return min(pending, key=lambda item: item[0]) if pending else None


def job_graph(job):
    return reachable(job[2], job[4] if len(job) > 4 else None)


def pending_encoders(pending, completed):
    """The next authoritative head's reachable, unfinished encoder attempts."""
    following = head(pending)
    if following is None:
        return
    graph = job_graph(following)
    for node_id, node, values in next_encoders(graph):
        token = (following[1], node_id, fingerprint(node))
        if token not in completed:
            yield following, graph, node_id, node, values, token

"""Read native execution progress independently of a connected browser client."""


def sampling_in(registry, prompt_id):
    if registry.prompt_id != prompt_id:
        return False
    for node_id, entry in registry.nodes.copy().items():
        state = getattr(entry.get("state"), "value", entry.get("state"))
        if state == "running":
            node = registry.dynprompt.get_node(node_id)
            if "Sampler" in node.get("class_type", ""):
                return True
    return False


def sampling_now(prompt_id, graph):
    from comfy_execution.progress import get_progress_state

    return sampling_in(get_progress_state(), prompt_id)

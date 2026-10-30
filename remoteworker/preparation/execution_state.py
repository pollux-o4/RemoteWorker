"""Read native execution progress independently of a connected browser client."""


from .._native import sampling_in


def sampling_now(prompt_id, graph):
    from comfy_execution.progress import get_progress_state

    return sampling_in(get_progress_state(), prompt_id)

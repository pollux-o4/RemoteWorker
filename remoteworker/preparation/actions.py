"""One-shot next conditioning and interruptible dummy VAE preparation."""

import time
from .gates import gate
from .invocation import call_values, speculative_call
from ..core.cancellation import cancellation_scope, check_cancelled, RequestCancelled


def prefetch(
    owner,
    current,
    following,
    graph,
    node_id,
    node,
    values,
    token,
    *,
    allow_after_sampling=False,
):
    url = values["server_url"]
    with gate(url).enter(speculative=True):
        if token in owner.done or owner.preparation_cancelled(
            current[1], following[1], graph
        ):
            return
        if not allow_after_sampling and not owner.eligible(current[1]):
            return
        if owner.scheduler:
            owner.scheduler.decide(
                current[1],
                following[1],
                needs_clip=True,
                decoder=owner.decoder(current),
            )
            owner.scheduler.invalidate_vae(current[1], url)
        owner.done.append(token)
        owner.record("conditioning", prompt_id=following[1], node_id=node_id, peer=url)
        started = time.monotonic()
        try:
            # Sampling completion does not cancel an admitted CLIP. Native decode waits at the same gate.
            with speculative_call(), cancellation_scope(
                lambda: owner.preparation_cancelled(current[1], following[1], graph),
                caller_interrupt=allow_after_sampling,
            ):
                result = owner.encoders[node["class_type"]]().run(**values)
                check_cancelled()
                if owner.handoffs:
                    from .handoff import pack, signature

                    normalized = call_values(
                        owner.encoders[node["class_type"]].run, values
                    )
                    blob = pack(result, max_bytes=owner.handoffs.limit)
                    check_cancelled()
                    stored = blob is not None and owner.handoffs.put(
                        (following[1], node_id), signature(normalized), blob
                    )
                    if not stored:
                        owner.record(
                            "skipped",
                            prompt_id=following[1],
                            node_id=node_id,
                            reason="handoff_too_large",
                        )
                    del blob, result
                else:
                    stored = True
                if owner.scheduler:
                    owner.scheduler.observe_encoder(time.monotonic() - started)
        except RequestCancelled:
            try:
                owner.done.remove(token)
            except ValueError:
                pass
            owner.record(
                "cancelled",
                prompt_id=following[1],
                node_id=node_id,
                reason="queue_changed",
            )
            raise
        except Exception:
            try:
                owner.done.remove(token)
            except ValueError:
                pass
            raise
        if stored:
            owner.record(
                "conditioning_ready", prompt_id=following[1], node_id=node_id, peer=url
            )


def prepare_vae(owner, current, decoder):
    if not decoder or not owner.eligible(current[1]):
        return
    node_id, url, model, kind = decoder
    token = (current[1], node_id, model)
    if owner.local_peer(url):
        return
    if owner.scheduler:
        if owner.scheduler.reserved(current[1], decoder):
            return
    elif token in owner.done:
        return
    if not owner.available(url):
        owner.record("deferred", prompt_id=current[1], reason="peer_busy")
        return
    with gate(url).enter(speculative=True):
        if not owner.eligible(current[1]) or owner.next_clip_missing(current[1], url):
            return
        owner.record(
            "vae",
            prompt_id=current[1],
            node_id=node_id,
            peer=url,
            model=model,
            total_steps=owner.scheduler.last.get("total") if owner.scheduler else None,
            remaining_steps=owner.scheduler.last.get("remaining_steps")
            if owner.scheduler
            else None,
        )
        started = time.monotonic()
        try:
            with cancellation_scope(
                lambda: (
                    not owner.prompt_present(current[1])
                    or owner.next_clip_missing(current[1], url)
                ),
                caller_interrupt=False,
            ):
                owner.warmup(url, model, kind)
                check_cancelled()
        except RequestCancelled:
            switched = owner.prompt_present(current[1]) and owner.next_clip_missing(
                current[1], url
            )
            owner.record(
                "vae_deferred" if switched else "cancelled",
                prompt_id=current[1],
                node_id=node_id,
                reason="next_clip_priority" if switched else "queue_changed",
            )
            raise
        owner.done.append(token)
        if owner.scheduler:
            owner.scheduler.observe_vae(time.monotonic() - started)
            owner.scheduler.reserve(current[1], decoder)
        owner.record(
            "vae_ready", prompt_id=current[1], node_id=node_id, peer=url, model=model
        )

"""Native cache-provider adapter around Rust-owned weak model reuse."""

from .._native import WeakModelCache, model_cache_eligible


class ModelReuse:
    def __init__(self):
        self.cache = WeakModelCache()

    @property
    def hits(self):
        return self.cache.hits

    def clear(self):
        self.cache.clear()

    def remember(self, key, outputs):
        self.cache.remember(key, outputs)

    def recall(self, key):
        return self.cache.recall(key)

    def should_cache(self, context, value=None):
        import nodes

        return model_cache_eligible(nodes.NODE_CLASS_MAPPINGS.get(context.class_type), value)

    async def on_store(self, context, value):
        self.remember((context.class_type, context.cache_key_hash), value.outputs)

    async def on_lookup(self, context):
        from comfy_execution.cache_provider import CacheValue

        outputs = self.recall((context.class_type, context.cache_key_hash))
        if outputs is not None:
            self.cache.hits += 1
            return CacheValue(outputs=outputs, ui={})
        return None

    def on_prompt_start(self, prompt_id):
        pass

    def on_prompt_end(self, prompt_id):
        pass

"""Weak model reuse across the two native executor caches; native signatures own invalidation."""
from collections import OrderedDict
import weakref


class ModelReuse:
    limit = 16

    def __init__(self):
        self.entries = OrderedDict()
        self.hits = 0

    def clear(self):
        self.entries.clear()

    def remember(self, key, outputs):
        def pack(value):
            if isinstance(value, (list, tuple)):
                return (type(value), [pack(part) for part in value])
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            return weakref.ref(value)
        try:
            packed = pack(outputs)
        except TypeError:
            return
        self.entries[key] = packed
        self.entries.move_to_end(key)
        while len(self.entries) > self.limit:
            self.entries.popitem(last=False)

    def recall(self, key):
        if key not in self.entries:
            return None
        def unpack(value):
            if isinstance(value, weakref.ReferenceType):
                result = value()
                if result is None:
                    raise ReferenceError()
                return result
            if isinstance(value, tuple):
                kind, parts = value
                return kind(unpack(part) for part in parts)
            return value
        try:
            return unpack(self.entries[key])
        except ReferenceError:
            self.entries.pop(key, None)
            return None

    def should_cache(self, context, value=None):
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get(context.class_type)
        return cls is not None and "MODEL" in getattr(cls, "RETURN_TYPES", ()) and not (value and value.ui)

    async def on_store(self, context, value):
        self.remember((context.class_type, context.cache_key_hash), value.outputs)

    async def on_lookup(self, context):
        from comfy_execution.cache_provider import CacheValue
        outputs = self.recall((context.class_type, context.cache_key_hash))
        if outputs is not None:
            self.hits += 1
            return CacheValue(outputs=outputs, ui={})
        return None

    def on_prompt_start(self, prompt_id):
        pass

    def on_prompt_end(self, prompt_id):
        pass

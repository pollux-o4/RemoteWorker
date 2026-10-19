"""Resolve audited CPU-only ancestors without touching Comfy's execution state."""

from ..core.cancellation import check_cancelled

SAFE_TYPES = frozenset(
    {
        "LoadImage",
        "GetImageSize",
        "ImageScaleToTotalPixels",
        "ImageScale",
        "ImageScaleBy",
        "ImageFromBatch",
        "ImageBatch",
        "PrimitiveFloat",
        "PrimitiveInt",
        "PrimitiveBoolean",
        "PrimitiveString",
        "PrimitiveStringMultiline",
        "RequestConditionInput",
        "ComfyMathExpression",
        "RemoteVAESelector",
    }
)


class UnsupportedInput(ValueError):
    pass


class Resolver:
    def __init__(self, graph, classes):
        self.graph, self.classes = graph, classes
        self.outputs, self.visiting = {}, set()

    def inputs(self, node_id, exclude=()):
        return {
            name: self.value(value)
            for name, value in self.graph[node_id].get("inputs", {}).items()
            if name not in exclude
        }

    def value(self, value):
        if not (
            isinstance(value, list) and len(value) == 2 and str(value[0]) in self.graph
        ):
            return value
        key, slot = str(value[0]), value[1]
        if key not in self.outputs:
            node = self.graph[key]
            kind = node["class_type"]
            if kind not in SAFE_TYPES or kind not in self.classes:
                raise UnsupportedInput("upstream_not_cpu_safe: " + kind)
            if key in self.visiting or len(self.visiting) > 64:
                raise UnsupportedInput("upstream_cycle_or_depth")
            self.visiting.add(key)
            try:
                check_cancelled()
                values = self.inputs(key)
                if kind == "ComfyMathExpression":
                    values = {
                        "expression": values["expression"],
                        "values": {
                            k.split(".", 1)[1]: v
                            for k, v in values.items()
                            if k.startswith("values.")
                        },
                    }
                cls = self.classes[kind]
                # Native GetImageSize also emits UI progress using per-execution hidden state.
                # Its CPU shape calculation is sufficient here; never mutate native class.hidden.
                if kind == "GetImageSize":
                    image = values["image"]
                    result = (image.shape[2], image.shape[1], image.shape[0])
                else:
                    result = getattr(cls(), getattr(cls, "FUNCTION", "execute"))(
                        **values
                    )
                if hasattr(result, "result"):
                    result = result.result
                if isinstance(result, dict):
                    if result.get("expand"):
                        raise UnsupportedInput("upstream_expansion")
                    result = result.get("result")
                if not isinstance(result, (tuple, list)):
                    raise UnsupportedInput("upstream_result: " + kind)
                self.outputs[key] = result
                check_cancelled()
            finally:
                self.visiting.remove(key)
        return self.outputs[key][slot]

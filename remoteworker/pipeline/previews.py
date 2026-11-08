"""CPU native previews, published before GPU reentry; final saving stays in the graph."""

from functools import wraps

from .. import _native


class Previews:
    def __init__(self, preview_class, classes, send, current_job, context):
        self.preview_class, self.classes, self.send = preview_class, classes, send
        self.current_job, self.context = current_job, context
        self.render = preview_class.save_images

    def install(self):
        @wraps(self.render)
        def save(instance, images, *args, **kwargs):
            job = self.current_job()
            context = self.context()
            ui = _native.pipeline_preview_cached(job, context)
            if ui is not None:
                return {"ui": ui, "result": (images,)}
            return self.render(instance, images, *args, **kwargs)

        self.preview_class.save_images = save

    def publish(self, job, dynprompt, decoder_id, result, extra, outputs, list_index=0):
        return _native.pipeline_preview_publish(self, job, dynprompt, decoder_id, result, extra, outputs, list_index or 0)

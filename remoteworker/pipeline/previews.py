"""CPU native previews, published before GPU reentry; final saving stays in the graph."""
from functools import wraps
import logging


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
            if job is not None and context is not None and context.prompt_id == job.prompt_id:
                ui = job.previews.pop((context.node_id, getattr(context, "list_index", 0) or 0), None)
                if ui is not None:
                    return {'ui': ui, 'result': (images,)}
            return self.render(instance, images, *args, **kwargs)
        self.preview_class.save_images = save

    def publish(self, job, dynprompt, decoder_id, result, extra, outputs, list_index=0):
        if extra.get("client_id") is None:
            return
        values = result.get('result', ()) if isinstance(result, dict) else result
        if not isinstance(values, (tuple, list)) or not values:
            return
        images = values[0]
        if getattr(getattr(images, 'device', None), 'type', None) != 'cpu' or getattr(images, 'ndim', None) != 4:
            return
        # Reachability preserves partial execution; bypassed/disabled output nodes stay untouched.
        reachable = set()
        def visit(key):
            if key in reachable:
                return
            reachable.add(key)
            for value in dynprompt.get_node(key).get('inputs', {}).values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int):
                    visit(value[0])
        for key in outputs:
            visit(key)
        for key in dynprompt.all_node_ids():
            node = dynprompt.get_node(key)
            if (key not in reachable or self.classes.get(node['class_type']) is not self.preview_class
                    or node['inputs'].get('images') != [decoder_id, 0]):
                continue
            if job.cancel.is_set():
                return
            try:
                # Native temp PNG contract only; no GPU work, execution globals or final SaveImage.
                rendered = self.render(self.preview_class(), images,
                    prompt=dynprompt.original_prompt, extra_pnginfo=extra.get('extra_pnginfo'))
                if job.cancel.is_set():
                    return
                ui = rendered['ui']
                job.previews[(key, list_index or 0)] = ui
                combined = {'images': [image for (node_id, _), value in job.previews.items()
                                       if node_id == key for image in value.get('images', [])]}
                self.send('executed', {'node': key, 'display_node': dynprompt.get_display_node_id(key),
                    'output': combined, 'prompt_id': job.prompt_id}, extra.get('client_id'))
                if key not in job.preview_nodes:
                    job.preview_nodes.append(key)
            except Exception:
                # Normal native execution remains authoritative and retries on GPU reentry.
                logging.getLogger(__name__).exception('Early preview failed; deferring to native output node')

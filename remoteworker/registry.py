"""ComfyUI node exports and explicit route registration."""
from .conditioning import service as conditioning
from .integration import artifact_io, discovery_routes, profile_routes, vae_selector
from .minimax import image_to_video, reference
from .prompt import service as prompt
from .qwen import service as qwen
from .qwen import worker as qwen_worker

NODE_MODULES = (artifact_io, conditioning, reference, image_to_video, prompt, vae_selector, qwen_worker, qwen)


def collect_nodes(modules):
    classes, labels = {}, {}
    for module in modules:
        node_classes = module.NODE_CLASS_MAPPINGS
        node_labels = module.NODE_DISPLAY_NAME_MAPPINGS
        duplicates = classes.keys() & node_classes.keys()
        if duplicates:
            raise ValueError(f"Duplicate ComfyUI node IDs: {sorted(duplicates)}")
        if node_classes.keys() != node_labels.keys():
            raise ValueError(f"Node display names do not match registered IDs in {module.__name__}")
        classes.update(node_classes)
        labels.update(node_labels)
    return classes, labels


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = collect_nodes(NODE_MODULES)


def register_routes():
    discovery_routes.register_routes()
    profile_routes.register_routes()

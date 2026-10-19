// Older API-imported graphs persisted empty optional combo values.
// Preserve explicit choices, including unknown values for backend validation.
const defaults = {peer_memory_policy: 'release_after_result', cache_policy: 'reuse_same_inputs'};
export function restorePolicyDefaults(node) {
  for (const widget of node.widgets || []) {
    if (Object.hasOwn(defaults, widget.name) && (widget.value == null || widget.value === ''))
      widget.value = defaults[widget.name];
  }
}

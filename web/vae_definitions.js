import { fetchJson, normalizeRemoteVaeOptions } from "./remote_choices.js";

export const VAE_NODE_TYPES = new Set(["RemoteVAEDecode", "RemoteVAEDecodeAudio", "RemoteVAEWarmup", "RemoteVAEVideo"]);
const advancedInputs = new Map();
const initialModels = new Map();
const readinessTypes = new Map();

export function vaeNodeType(node) {
    return node.comfyClass || node.constructor?.comfyClass || node.type;
}

export function readinessType(node) {
    return readinessTypes.get(vaeNodeType(node));
}

export function recordVaeDefinition(definition) {
    if (!VAE_NODE_TYPES.has(definition?.name)) return;
    const fields = { ...definition.input?.required, ...definition.input?.optional };
    const signal = definition.name === "RemoteVAEWarmup" ? definition.output?.[0] : fields.warmup?.[0];
    if (signal) readinessTypes.set(definition.name, signal);
    advancedInputs.set(definition.name, new Set(Object.entries(fields)
        .filter(([, config]) => config[1]?.advanced).map(([name]) => name)));
}

export function applyAdvancedInputs(node) {
    const type = vaeNodeType(node);
    for (const widget of node.widgets || []) {
        if (advancedInputs.get(type)?.has(widget.name)) {
            widget.advanced = true;
            widget.hidden = !node.showAdvanced;
        }
    }
}

export function applyInitialVaeDefinitions(definitions) {
    for (const definition of Object.values(definitions || {})) {
        if (!VAE_NODE_TYPES.has(definition?.name)) continue;
        const fields = definition.input?.required;
        const names = initialModels.get(fields?.server_url?.[1]?.default);
        if (names) fields.vae_name[0] = ["", ...names];
        recordVaeDefinition(definition);
    }
}

export async function preloadVaeDefinitions(definitions) {
    const targets = Object.values(definitions || {}).filter((definition) => VAE_NODE_TYPES.has(definition?.name));
    const servers = new Set(targets.map((definition) => definition.input?.required?.server_url?.[1]?.default).filter(Boolean));
    await Promise.all([...servers].map(async (server_url) => {
        try {
            const payload = await fetchJson("/remote-conditioning/vae-loader-options", { server_url, timeout_seconds: 5 });
            initialModels.set(server_url, normalizeRemoteVaeOptions(payload).vae_name);
        } catch {
            initialModels.delete(server_url);
        }
    }));
    applyInitialVaeDefinitions(definitions);
}

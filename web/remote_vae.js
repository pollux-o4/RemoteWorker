import { setupDelivery, applyDelivery, migrateDelivery } from "./vae_delivery.js";
import { app } from "../../scripts/app.js";
import { applyChoiceWidget, normalizeRemoteVaeOptions, refreshRemoteChoiceOptions } from "./remote_choices.js";

import { applyAdvancedInputs, recordVaeDefinition, preloadVaeDefinitions, applyInitialVaeDefinitions, VAE_NODE_TYPES, vaeNodeType, readinessType } from "./vae_definitions.js";


function modelWidget(node) {
    return node.widgets?.find((widget) => widget.name === "vae_name");
}

export function applyVaeModels(node, payload) {
    const names = normalizeRemoteVaeOptions(payload).vae_name;
    const widget = modelWidget(node);
    if (!widget) throw new Error("VAE 선택 항목 없음");
    // Never switch a workflow to another model as a side effect of discovery.
    applyChoiceWidget(widget, names);
    const audio = node.widgets?.find((item) => item.name === "audio_vae_name");
    if (audio) applyChoiceWidget(audio, ["", ...names]);
    node._remoteVaeModels = names;
    app.graph?.setDirtyCanvas?.(true, true);
    return names.length;
}

export function modelStatus(node) {
    const names = node._remoteVaeModels;
    if (!names) return "VAE 새로고침";
    const selected = modelWidget(node)?.value;
    if (!selected) return "VAE 선택 필요 · 새로고침";
    const audio = node.widgets?.find(item => item.name === "audio_vae_name")?.value;
    if (!names.includes(selected) || (audio && !names.includes(audio))) return "VAE 없음 · 다시 선택";
    return `VAE ${names.length}개 · 새로고침`;
}

export async function refreshVaeModels(node, button, notify = true) {
    return refreshRemoteChoiceOptions(node, button, {
        loadingKey: "_remoteVaeLoading",
        setLabel: (label) => { button.label = label; },
        path: "/remote-conditioning/vae-loader-options",
        widgetNames: ["vae_name", "audio_vae_name"],
        missingSettingsLabel: "서버 설정 확인",
        loadingLabel: "VAE 조회 중…",
        serverChangedLabel: "서버 변경됨 · 새로고침",
        successLabel: () => modelStatus(node),
        notify,
        failureLabel: () => "조회 실패 · VAE 재시도",
        warningPrefix: "[Remote VAE] Model discovery failed",
        apply: applyVaeModels,
    });
}

export function normalizeVaeSockets(node) {
    const type = vaeNodeType(node);
    if (!VAE_NODE_TYPES.has(type)) return;
    const signalType = readinessType(node);
    if (!signalType) return;
    if (type === "RemoteVAEWarmup" && node.outputs?.[0]) node.outputs[0].type = signalType;
    const input = node.inputs?.find((slot) => slot.name === "warmup");
    if (input) input.type = signalType;
}

export function resizeVaeNode(node) {
    const computed = node.computeSize?.();
    node.setSize?.([node.size?.[0] || computed?.[0] || 240, computed?.[1] || node.size?.[1] || 160]);
    app.graph?.setDirtyCanvas?.(true, true);
}

export function setupVaeNode(node) {
    const type = vaeNodeType(node);
    if (!VAE_NODE_TYPES.has(type)) return;
    normalizeVaeSockets(node);
    applyAdvancedInputs(node);
    if (node._remoteVaeRefresh) {
        if (node._remoteVaeDetails) node._remoteVaeDetails.label = node.showAdvanced ? "고급 접기" : "고급 설정";
        return;
    }
    const button = node.addWidget("button", "remote_vae_models", null,
        () => refreshVaeModels(node, button), { serialize: false });
    button.label = "VAE 새로고침";
    button.serialize = false;
    node._remoteVaeRefresh = button;
    const details = node.addWidget("button", "remote_vae_details", null, () => {
        node.showAdvanced = !node.showAdvanced;
        applyAdvancedInputs(node);
        applyDelivery(node);
        details.label = node.showAdvanced ? "고급 접기" : "고급 설정";
        resizeVaeNode(node);
    }, { serialize: false });
    details.label = node.showAdvanced ? "고급 접기" : "고급 설정";
    details.serialize = false;
    node._remoteVaeDetails = details;
    const vae = modelWidget(node);
    if (vae) {
        const previous = vae.callback;
        vae.callback = function () {
            const result = previous?.apply(this, arguments);
            button.label = modelStatus(node);
            return result;
        };
    }
    const server = node.widgets?.find((widget) => widget.name === "server_url");
    if (server) {
        const previous = server.callback;
        server.callback = function () {
            const result = previous?.apply(this, arguments);
            delete node._remoteVaeModels;
            button.label = "서버 변경됨 · 새로고침";
            app.graph?.setDirtyCanvas?.(true, true);
            return result;
        };
    }
    resizeVaeNode(node);
    // Run after widget values are restored while loading saved workflows.
    Promise.resolve().then(() => refreshVaeModels(node, button, false)).finally(() => resizeVaeNode(node));
}

app.registerExtension({
    name: "RemoteWorker.VaeControls",
    addCustomNodeDefs: preloadVaeDefinitions,
    beforeRegisterVueAppNodeDefs: applyInitialVaeDefinitions,
    beforeRegisterNodeDef(_nodeType, nodeData) { recordVaeDefinition(nodeData); },
    beforeConfigureGraph: migrateDelivery,
    nodeCreated(node) { setupVaeNode(node); setupDelivery(node); },
    loadedGraphNode(node) {
        normalizeVaeSockets(node);
        setupVaeNode(node);
        setupDelivery(node);
    },
});

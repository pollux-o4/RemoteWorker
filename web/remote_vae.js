import { app } from "../../scripts/app.js";
import { normalizeRemoteVaeOptions, refreshRemoteChoiceOptions } from "./remote_choices.js";

import { applyAdvancedInputs, recordVaeDefinition, preloadVaeDefinitions, applyInitialVaeDefinitions, VAE_NODE_TYPES, vaeNodeType, readinessType } from "./vae_definitions.js";


function modelWidget(node) {
    return node.widgets?.find((widget) => widget.name === "vae_name");
}

export function applyVaeModels(node, payload) {
    const names = normalizeRemoteVaeOptions(payload).vae_name;
    const widget = modelWidget(node);
    if (!widget) throw new Error("VAE 선택 항목 없음");
    // Never switch a workflow to another model as a side effect of discovery.
    const selected = widget.value || "";
    widget.options = { ...(widget.options || {}), values: [...new Set([selected, ...names])] };
    node._remoteVaeModels = names;
    app.graph?.setDirtyCanvas?.(true, true);
    return names.length;
}

export function modelStatus(node) {
    const names = node._remoteVaeModels;
    if (!names) return "모델 목록 새로고침";
    const selected = modelWidget(node)?.value;
    if (!selected) return `모델 ${names.length}개 · VAE 선택 필요`;
    if (!names.includes(selected)) return "선택한 VAE 없음 · 다시 조회";
    return `모델 ${names.length}개 · 새로고침`;
}

export async function refreshVaeModels(node, button, notify = true) {
    return refreshRemoteChoiceOptions(node, button, {
        loadingKey: "_remoteVaeLoading",
        setLabel: (label) => { button.label = label; },
        path: "/remote-conditioning/vae-loader-options",
        missingSettingsLabel: "서버 설정 확인 필요",
        loadingLabel: "모델 목록 확인 중…",
        serverChangedLabel: "서버 변경됨 · 다시 조회",
        successLabel: () => modelStatus(node),
        failureLabel: (detail) => {
            if (notify) app.extensionManager?.toast?.add({
                severity: "error", summary: "VAE 모델 조회 실패",
                detail: `서버 주소와 실행 상태를 확인하세요. ${detail.slice(0, 180)}`, life: 7000,
            });
            return "연결 실패 · 다시 조회";
        },
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
    node.setSize?.([Math.max(node.size?.[0] || 0, 360), node.computeSize?.()[1] || node.size?.[1] || 160]);
    app.graph?.setDirtyCanvas?.(true, true);
}

export function setupVaeNode(node) {
    const type = vaeNodeType(node);
    if (!VAE_NODE_TYPES.has(type)) return;
    normalizeVaeSockets(node);
    applyAdvancedInputs(node);
    if (node._remoteVaeRefresh) {
        if (node._remoteVaeDetails) node._remoteVaeDetails.label = node.showAdvanced ? "고급 설정 숨기기" : "고급 설정 표시";
        return;
    }
    const button = node.addWidget("button", "remote_vae_models", null,
        () => refreshVaeModels(node, button), { serialize: false });
    button.label = "모델 목록 새로고침";
    button.serialize = false;
    node._remoteVaeRefresh = button;
    const details = node.addWidget("button", "remote_vae_details", null, () => {
        node.showAdvanced = !node.showAdvanced;
        applyAdvancedInputs(node);
        details.label = node.showAdvanced ? "고급 설정 숨기기" : "고급 설정 표시";
        resizeVaeNode(node);
    }, { serialize: false });
    details.label = node.showAdvanced ? "고급 설정 숨기기" : "고급 설정 표시";
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
            button.label = "서버 변경됨 · 모델 목록 조회";
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
    nodeCreated: setupVaeNode,
    loadedGraphNode(node) {
        normalizeVaeSockets(node);
        setupVaeNode(node);
    },
});

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { applyDelivery } from "./vae_delivery.js";

const TYPE = "RemoteVAECombineMedia";
const get = (node, name) => node.widgets?.find((widget) => widget.name === name);

export function sharedSources(node) {
    if ((node.comfyClass || node.type) !== "CreateVideo") return null;
    const inputs = ["images", "audio"].map((name) => {
        const slot = node.inputs?.find((input) => input.name === name);
        const link = node.graph?.links?.[slot?.link];
        return link ? node.graph.getNodeById(link.origin_id) : null;
    });
    if ((inputs[0]?.comfyClass || inputs[0]?.type) !== "RemoteVAEDecode") return null;
    if (inputs[1] && (inputs[1].comfyClass || inputs[1].type) !== "RemoteVAEDecodeAudio") return null;
    return inputs;
}

export function convertSharedVideo(node) {
    const sources = sharedSources(node);
    if (!sources) throw new Error("원격 VAE의 공유 모드 연결을 먼저 확인하세요.");
    if (![undefined, "auto", 8, "8"].includes(get(node, "bit_depth")?.value)
        || ![undefined, "sRGB"].includes(get(node, "color_space")?.value)
        || node.inputs?.find((input) => input.name === "fps")?.link != null) {
        throw new Error("공유 파일 결합은 sRGB 8비트와 고정 프레임 속도를 지원합니다. 현재 설정은 텐서 연결을 유지하세요.");
    }
    const graph = node.graph;
    const replacement = globalThis.LiteGraph.createNode(TYPE);
    if (!replacement) throw new Error("서버에 파일 결합 노드가 없습니다. 서버 업데이트 후 다시 연결하세요.");
    const downstream = (node.outputs?.[0]?.links || []).map((id) => graph.links[id]).filter(Boolean)
        .map((link) => [graph.getNodeById(link.target_id), link.target_slot]);
    graph.beforeChange?.();
    try {
        replacement.pos = [...node.pos];
        graph.add(replacement);
        sources.forEach((source, index) => {
            if (!source) return;
            const mode = get(source, "output_mode");
            mode.value = "shared";
            mode.callback?.("shared");
            if (index === 0 && get(node, "fps") && get(source, "fps")) get(source, "fps").value = get(node, "fps").value;
            source.connect(1, replacement, index);
        });
        for (const [target, slot] of downstream) if (target) replacement.connect(0, target, slot);
        graph.remove(node);
        sources.filter(Boolean).forEach(applyDelivery);
        graph.setDirtyCanvas?.(true, true);
    } finally {
        graph.afterChange?.();
    }
    return replacement;
}

function setupMigration(node) {
    if ((node.comfyClass || node.type) !== "CreateVideo" || node._sharedMigration) return;
    node._sharedMigration = true;
    const button = node.addWidget("button", "공유 파일 연결로 전환", null, () => {
        try { convertSharedVideo(node); }
        catch (error) { app.extensionManager?.toast?.add({severity: "error", summary: "연결 확인 필요", detail: error.message, life: 6000}); }
    }, { serialize: false });
    button.serialize = false;
    const update = () => { button.hidden = !sharedSources(node); };
    const previous = node.onConnectionsChange;
    node.onConnectionsChange = function () { previous?.apply(this, arguments); update(); };
    update();
}

function setupResult(node) {
    if ((node.comfyClass || node.type) !== TYPE || node._sharedResult) return;
    node._sharedResult = true;
    const panel = document.createElement("section");
    panel.className = "remote-vae-delivery";
    const status = document.createElement("p");
    status.setAttribute("role", "status");
    status.textContent = "영상·음성의 파일 경로를 연결하세요.";
    panel.append(status);
    const previous = node.onExecuted;
    node.onExecuted = function (message) {
        previous?.apply(this, arguments);
        const result = message?.shared_media?.find((item) => item.final);
        if (!result) return;
        panel.querySelector("video")?.pause();
        const video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.src = api.apiURL(result.url);
        video.setAttribute("aria-label", "완성 영상");
        const link = document.createElement("a");
        link.href = video.src; link.textContent = "완성 영상 열기 ↗";
        link.target = "_blank"; link.rel = "noopener";
        status.textContent = message.source_cleanup?.[0] ? "완료 · 중간 파일 정리됨" : "완료 · 중간 파일 보존";
        panel.replaceChildren(status, video, link);
        if (message.source_cleanup?.[0]) document.dispatchEvent(new CustomEvent("remote-vae:finalized", {detail: result}));
        node.setSize?.([Math.max(node.size?.[0] || 0, 360), node.computeSize()[1]]);
    };
    const widget = node.addDOMWidget("shared_video_result", "custom", panel, {serialize: false});
    widget.serialize = false;
    widget.computeSize = (width) => [width, panel.querySelector("video") ? 260 : 60];
    node.setSize?.([360, node.computeSize()[1]]);
}

if (typeof document !== "undefined" && !document.getElementById("remote-vae-delivery-style")) {
    const link = document.createElement("link");
    link.id = "remote-vae-delivery-style"; link.rel = "stylesheet";
    link.href = new URL("./vae_delivery.css", import.meta.url).href;
    document.head.append(link);
}
app.registerExtension({
    name: "RemoteWorker.SharedVideo",
    nodeCreated(node) { setupMigration(node); setupResult(node); },
    loadedGraphNode(node) { setupMigration(node); setupResult(node); },
});

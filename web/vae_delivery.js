import { api } from "../../scripts/api.js";

if (typeof document !== "undefined" && !document.getElementById("remote-vae-delivery-style")) {
    const link = document.createElement("link");
    link.id = "remote-vae-delivery-style";
    link.rel = "stylesheet";
    link.href = new URL("./vae_delivery.css", import.meta.url).href;
    document.head.append(link);
}

const TYPES = new Set(["RemoteVAEDecode", "RemoteVAEDecodeAudio"]);
const VERSION = "remoteVaeDelivery";
const value = (node, name) => node.widgets?.find((w) => w.name === name)?.value;

export function migrateDelivery(graph) {
    for (const node of graph?.nodes || []) {
        if (!TYPES.has(node.type) || node.properties?.[VERSION] === 1) continue;
        const audio = node.type === "RemoteVAEDecodeAudio";
        node.widgets_values ||= [];
        // Existing positional widgets stay in their original slots.
        const legacyCount = audio ? 4 : 9;
        if (Array.isArray(node.widgets_values) && !["shared", "tensor"].includes(node.widgets_values[legacyCount])) {
            const defaults = ["", "", 600, "release_after_result", false, 512, 64, 64, 8].slice(0, legacyCount);
            node.widgets_values = defaults.map((fallback, index) => node.widgets_values[index] ?? fallback);
            node.widgets_values.push("tensor", ...(audio ? [] : [24, "", 18]), "");
        }
        if (!Array.isArray(node.widgets_values)) node.widgets_values.output_mode ??= "tensor";
        if (node.outputs && node.outputs.length < 2) node.outputs.push({ name: "file_path", type: "STRING", links: null });
        if (node.widgets_values_named) node.widgets_values_named.output_mode ??= "tensor";
        node.properties = { ...node.properties, [VERSION]: 1 };
    }
    for (const subgraph of graph?.definitions?.subgraphs || []) migrateDelivery(subgraph);
}

export function deliveryMode(node) {
    return value(node, "output_mode") || "tensor";
}

export function applyDelivery(node) {
    if (!TYPES.has(node.comfyClass || node.type)) return;
    const shared = deliveryMode(node) === "shared";
    for (const widget of node.widgets || []) {
        if (["fps", "audio_vae_name", "crf", "request_id"].includes(widget.name)) {
            widget.hidden = !shared || (!!widget.advanced && !node.showAdvanced);
        }
        if (["tile_size", "overlap", "temporal_size", "temporal_overlap"].includes(widget.name)) widget.hidden = shared || !node.showAdvanced;
    }
    if (node.outputs?.[0]) node.outputs[0].label = shared ? "텐서 · 비활성" : (node.type === "RemoteVAEDecodeAudio" ? "오디오" : "이미지");
    if (node.outputs?.[1]) node.outputs[1].label = shared ? "파일 경로" : "파일 경로 · 비활성";
    const panel = node._deliveryPanel;
    if (panel) {
        panel.hidden = !shared;
        const linked = !!node.outputs?.[0]?.links?.length;
        panel.querySelector("[data-hint]").textContent = linked
            ? "공유 파일로 저장 · 기존 텐서 연결의 후속 분기는 건너뜁니다."
            : "처리 서버에 파일 저장 · 파일 경로 출력 사용";
    }
    node.setSize?.([Math.max(node.size?.[0] || 0, 390), node.computeSize?.()[1] || 260]);
    node.graph?.setDirtyCanvas?.(true, true);
}

export async function checkDelivery(node) {
    const status = node._deliveryPanel?.querySelector("[role=status]");
    if (!status) return;
    const server_url = value(node, "server_url");
    const generation = node._deliveryCheck = (node._deliveryCheck || 0) + 1;
    status.textContent = "양쪽 공유 폴더 확인 중…";
    status.dataset.state = "busy";
    try {
        const response = await api.fetchApi("/remote-vae/preflight", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ server_url }),
        });
        const result = await response.json();
        if (generation !== node._deliveryCheck || server_url !== value(node, "server_url")) return;
        if (!response.ok) throw new Error(result.error || "공유 폴더 확인 실패");
        status.textContent = "공유 폴더 연결됨";
        status.title = result.root;
        status.dataset.state = "ready";
    } catch (error) {
        if (generation !== node._deliveryCheck || server_url !== value(node, "server_url")) return;
        status.textContent = "공유 폴더 연결 실패 · 다시 확인";
        status.title = error.message;
        status.dataset.state = "error";
    }
}

function showResult(node, message) {
    const result = message?.shared_media?.[0];
    if (!result || !node._deliveryPanel) return;
    node._deliveryResult = result;
    const panel = node._deliveryPanel;
    node._deliveryCheck = (node._deliveryCheck || 0) + 1;
    const status = panel.querySelector("[role=status]");
    status.textContent = "파일 저장 완료";
    status.dataset.state = "ready";
    status.title = result.path;
    const target = panel.querySelector("[data-result]");
    target.replaceChildren();
    const media = document.createElement(result.kind === "audio" ? "audio" : "video");
    media.src = api.apiURL(result.url);
    media.controls = true;
    media.preload = "metadata";
    media.setAttribute("aria-label", "저장된 VAE 결과");
    const link = document.createElement("a");
    link.href = media.src;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "저장 파일 열기 ↗";
    const path = document.createElement("input");
    path.value = result.path;
    path.readOnly = true;
    path.setAttribute("aria-label", "최종 파일 경로");
    path.addEventListener("focus", () => path.select());
    target.append(media, link, path);
    node.setSize?.([Math.max(node.size?.[0] || 0, 390), node.computeSize?.()[1] || 400]);
}

export function setupDelivery(node) {
    if (!TYPES.has(node.comfyClass || node.type)) return;
    node.properties ||= {};
    node.properties[VERSION] = 1;
    if (node._deliverySetup) return applyDelivery(node);
    node._deliverySetup = true;
    if (typeof document !== "undefined" && node.addDOMWidget) {
        const panel = document.createElement("section");
        panel.className = "remote-vae-delivery";
        panel.innerHTML = '<p data-hint></p><div class="remote-vae-check"><span role="status" aria-live="polite">공유 폴더 확인 필요</span><button type="button">연결 확인</button></div><div data-result></div>';
        panel.querySelector("button").addEventListener("click", () => checkDelivery(node));
        node._deliveryPanel = panel;
        const dom = node.addDOMWidget("remote_vae_delivery", "custom", panel, { serialize: false, hideOnZoom: false });
        dom.serialize = false;
        dom.computeSize = (width) => [width, panel.hidden ? 0 : panel.querySelector("video") ? 360 : panel.querySelector("audio") ? 220 : 100];
        const finalized = (event) => {
            if (!event.detail.sources?.includes(node._deliveryResult?.request_id)) return;
            panel.querySelector("video, audio")?.pause();
            const link = document.createElement("a");
            link.href = api.apiURL(event.detail.url);
            link.target = "_blank"; link.rel = "noopener";
            link.textContent = "완성 영상 열기 ↗";
            panel.querySelector("[data-result]").replaceChildren(link);
            panel.querySelector("[role=status]").textContent = "완성 영상에 포함됨 · 중간 파일 정리됨";
            node.setSize?.([node.size[0], node.computeSize()[1]]);
        };
        document.addEventListener("remote-vae:finalized", finalized);
        const removed = node.onRemoved;
        node.onRemoved = function () {
            document.removeEventListener("remote-vae:finalized", finalized);
            return removed?.apply(this, arguments);
        };
    }
    for (const widget of node.widgets || []) {
        if (!["output_mode", "server_url"].includes(widget.name)) continue;
        const previous = widget.callback;
        widget.callback = function () {
            const result = previous?.apply(this, arguments);
            if (widget.name === "server_url") {
                const status = node._deliveryPanel?.querySelector("[role=status]");
                if (status) { status.textContent = "공유 폴더 확인 필요"; status.dataset.state = ""; status.title = ""; }
            }
            applyDelivery(node);
            return result;
        };
    }
    const previous = node.onExecuted;
    node.onExecuted = function (message) { previous?.apply(this, arguments); showResult(node, message); };
    applyDelivery(node);
}

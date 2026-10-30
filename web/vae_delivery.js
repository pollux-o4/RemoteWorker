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

function resizeDelivery(node) {
    if (!node.computeSize || !node.setSize || node._deliveryRemoved) return;
    const width = node.size?.[0] || 320;
    const height = node.computeSize()[1];
    if (height !== node.size?.[1]) node.setSize([width, height]);
    node.graph?.setDirtyCanvas?.(true, true);
}

function setDeliveryState(node, state, message, detail = "", next = "") {
    const panel = node._deliveryPanel;
    const status = panel?.querySelector("[role=status]");
    if (!status) return;
    status.textContent = message;
    status.dataset.state = state;
    status.title = "";
    const followup = panel.querySelector("[data-next]");
    if (followup && followup !== status) { followup.textContent = next; followup.hidden = !next; }
    const details = panel.querySelector("details");
    const diagnostic = panel.querySelector("[data-diagnostic]");
    if (details && details !== status) details.hidden = !detail;
    if (diagnostic && diagnostic !== status) diagnostic.textContent = detail;
    const button = panel.querySelector("button");
    if (button && button !== status) button.textContent = state === "busy" ? "확인 취소" : state === "error" ? "다시 확인" : "연결 확인";
    panel.setAttribute?.("aria-busy", String(state === "busy"));
    resizeDelivery(node);
}

function invalidateDelivery(node, message = "공유 폴더 확인 필요") {
    node._deliveryCheck = (node._deliveryCheck || 0) + 1;
    node._deliveryController?.abort();
    node._deliveryController = null;
    setDeliveryState(node, "idle", message);
}

export function applyDelivery(node) {
    if (!TYPES.has(node.comfyClass || node.type)) return;
    const shared = deliveryMode(node) === "shared";
    for (const widget of node.widgets || []) {
        if (["fps", "audio_vae_name", "crf", "request_id"].includes(widget.name)) {
            widget.hidden = widget.name === "request_id" ? (!shared || !node.showAdvanced) : true;
        }
        if (["tile_size", "overlap", "temporal_size", "temporal_overlap"].includes(widget.name)) widget.hidden = !node.showAdvanced;
    }
    if (node.outputs?.[0]) node.outputs[0].label = (node.comfyClass || node.type) === "RemoteVAEDecodeAudio" ? "오디오" : "이미지";
    if (node.outputs?.[1]) {
        node.outputs[1].label = "미사용";
        node.outputs[1].tooltip = "기존 워크플로 호환용 비활성 파일 경로 출력";
    }
    if (node._deliveryPanel) {
        node._deliveryPanel.hidden = !shared;
        if (!shared && node._deliveryController) invalidateDelivery(node);
    }
    resizeDelivery(node);
}

export async function checkDelivery(node) {
    if (!node._deliveryPanel?.querySelector("[role=status]") || node._deliveryRemoved) return;
    const server_url = value(node, "server_url");
    node._deliveryController?.abort();
    const generation = node._deliveryCheck = (node._deliveryCheck || 0) + 1;
    const controller = new AbortController();
    node._deliveryController = controller;
    const current = () => generation === node._deliveryCheck && !node._deliveryRemoved && server_url === value(node, "server_url");
    if (!server_url?.trim()) {
        node._deliveryController = null;
        setDeliveryState(node, "error", "서버 주소 입력 필요");
        return;
    }
    setDeliveryState(node, "busy", "연결 확인 중");
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 10000);
    try {
        const response = await api.fetchApi("/remote-vae/preflight", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ server_url }), signal: controller.signal,
        });
        const result = await response.json();
        if (!current()) return;
        if (!response.ok) throw new Error(result?.error || "공유 폴더 확인 실패");
        if (typeof result?.root !== "string" || !result.root.trim() || result.ready !== true) {
            throw new Error("공유 폴더 확인 결과가 없습니다. 다시 확인하세요.");
        }
        setDeliveryState(node, "ready", "공유 폴더 연결됨", result.root || "");
    } catch (error) {
        if (!current()) return;
        setDeliveryState(node, "error", timedOut ? "연결 응답 지연" : "공유 폴더 연결 실패", error.message);
    } finally {
        clearTimeout(timeout);
        if (generation === node._deliveryCheck) node._deliveryController = null;
    }
}

function showResult(node, message) {
    if (!Array.isArray(message?.shared_transfer) || !message.shared_transfer.length || !node._deliveryPanel || node._deliveryRemoved) return;
    invalidateDelivery(node);
    setDeliveryState(node, "ready", "전송 완료", "", "저장은 후속 노드에서");
}

export function setupDelivery(node) {
    if (!TYPES.has(node.comfyClass || node.type)) return;
    node.properties ||= {};
    node.properties[VERSION] = 1;
    if (node._deliverySetup) return applyDelivery(node);
    node._deliverySetup = true;
    if (typeof document !== "undefined" && node.addDOMWidget) {
        const panel = document.createElement("section"); panel.className = "remote-vae-delivery";
        const row = document.createElement("div"); row.className = "remote-vae-check";
        const status = document.createElement("span"); status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
        status.textContent = "공유 폴더 확인 필요";
        const button = document.createElement("button"); button.type = "button"; button.textContent = "연결 확인";
        button.addEventListener("click", () => node._deliveryController
            ? invalidateDelivery(node, "확인 취소됨") : checkDelivery(node));
        row.append(status, button);
        const next = document.createElement("p"); next.setAttribute("data-next", ""); next.hidden = true;
        const details = document.createElement("details"); details.hidden = true;
        const summary = document.createElement("summary"); summary.textContent = "상세";
        const diagnostic = document.createElement("p"); diagnostic.setAttribute("data-diagnostic", "");
        diagnostic.setAttribute("tabindex", "0"); diagnostic.setAttribute("role", "region");
        diagnostic.setAttribute("aria-label", "공유 폴더 연결 상세");
        details.append(summary, diagnostic);
        details.addEventListener("toggle", () => resizeDelivery(node));
        panel.append(row, next, details);
        node._deliveryPanel = panel;
        const dom = node.addDOMWidget("remote_vae_delivery", "custom", panel, { serialize: false, hideOnZoom: false });
        dom.serialize = false;
        dom.computeSize = (width) => [width, panel.hidden ? 0 : Math.max(76, Math.ceil(Math.max(panel.scrollHeight || 0, panel.offsetHeight || 0)))];
        if (typeof ResizeObserver !== "undefined") {
            node._deliveryObserver = new ResizeObserver(() => resizeDelivery(node));
            node._deliveryObserver.observe(panel);
        }
    }
    for (const widget of node.widgets || []) {
        if (!["output_mode", "server_url"].includes(widget.name)) continue;
        const previous = widget.callback;
        widget.callback = function () {
            const result = previous?.apply(this, arguments);
            if (widget.name === "server_url" || deliveryMode(node) !== "shared") invalidateDelivery(node);
            applyDelivery(node);
            return result;
        };
    }
    const previous = node.onExecuted;
    node.onExecuted = function (message) { previous?.apply(this, arguments); showResult(node, message); };
    const removed = node.onRemoved;
    node.onRemoved = function () {
        node._deliveryRemoved = true;
        node._deliveryCheck = (node._deliveryCheck || 0) + 1;
        node._deliveryController?.abort(); node._deliveryController = null;
        node._deliveryObserver?.disconnect();
        return removed?.apply(this, arguments);
    };
    const connections = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const result = connections?.apply(this, arguments);
        queueMicrotask(() => applyDelivery(node));
        return result;
    };
    applyDelivery(node);
}

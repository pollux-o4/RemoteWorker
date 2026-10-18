import { app } from "../../scripts/app.js";
import { cleanOptions, refreshRemoteChoiceOptions } from "./remote_choices.js";

const QWEN_NODES = new Set(["RemoteQwenImage21"]);

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function placeAfter(node, item, anchorName) {
    const widgets = node.widgets || [];
    const current = widgets.indexOf(item);
    const anchor = widgets.findIndex((candidate) => candidate.name === anchorName);
    if (current < 0 || anchor < 0 || current === anchor + 1) return;
    widgets.splice(current, 1);
    widgets.splice(anchor + 1, 0, item);
}

export async function refreshPeerChoices(node, button, kind) {
    const encoder = kind === "encoder";
    const label = encoder ? "텍스트 인코더" : "VAE";
    const name = encoder ? "remote_text_encoder" : "vae_name";
    return refreshRemoteChoiceOptions(node, button, {
        loadingKey: "_remoteQwenRefreshInFlight",
        path: encoder ? "/remote-conditioning/clip-loader-options" : "/remote-conditioning/vae-loader-options",
        missingSettingsLabel: "⚠ 원격 주소를 먼저 입력하세요",
        loadingLabel: `… 원격 ${label} 조회 중`,
        serverChangedLabel: "↻ 원격 주소 변경됨 — 다시 조회",
        successLabel: (count) => `✓ ${label} ${count}개 — 다시 조회`,
        failureLabel: (detail) => `⚠ 조회 실패: ${detail}`,
        warningPrefix: "[Remote Qwen Image 2.1] peer model lookup failed",
        apply(targetNode, payload) {
            const choices = cleanOptions(encoder ? payload?.clip_name : payload?.vae_name, name);
            const target = widget(targetNode, name);
            if (!target) throw new Error(`${name} 선택 항목을 찾을 수 없습니다`);
            target.options = { ...(target.options || {}), values: choices };
            target.type = "combo";
            if (!choices.includes(target.value)) target.value = choices[0];
            target.callback?.(target.value, app.canvas, targetNode, app.canvas?.graph_mouse);
            return choices.length;
        },
    });
}

app.registerExtension({
    name: "RemoteQwenImage21.PeerChoiceRefresh",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!QWEN_NODES.has(nodeData?.name)) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const encoderButton = this.addWidget(
                "button",
                "↻ 이 주소의 텍스트 인코더 목록 새로고침",
                "",
                () => refreshPeerChoices(this, encoderButton, "encoder"),
                { serialize: false },
            );
            const vaeButton = this.addWidget(
                "button",
                "↻ 이 주소의 참조 이미지 VAE 목록 새로고침",
                "",
                () => refreshPeerChoices(this, vaeButton, "vae"),
                { serialize: false },
            );
            placeAfter(this, encoderButton, "remote_text_encoder");
            placeAfter(this, vaeButton, "vae_name");
            this.setSize?.(this.computeSize?.() || this.size);
            app.graph?.setDirtyCanvas?.(true, true);
            return result;
        };
    },
});

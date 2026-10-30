import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "RemoteVAECombineMedia";
// Cached frontend callbacks must not connect the decoder's inactive legacy STRING output.
export function convertSharedVideo() {
    throw new Error("파일 경로는 RemoteVAEVideo에서 연결하세요.");
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
    nodeCreated(node) { setupResult(node); },
    loadedGraphNode(node) { setupResult(node); },
});

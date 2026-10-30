import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "RemoteVAECombineMedia";
function releaseVideo(video) {
    video?.pause(); video?.removeAttribute?.("src"); video?.load?.();
}
// Cached frontend callbacks must not connect the decoder's inactive legacy STRING output.
export function convertSharedVideo() {
    throw new Error("파일 경로는 RemoteVAEVideo에서 연결하세요.");
}

export function setupResult(node) {
    if ((node.comfyClass || node.type) !== TYPE || node._sharedResult) return;
    node._sharedResult = true;
    const panel = document.createElement("section"); panel.className = "remote-vae-delivery";
    const status = document.createElement("p"); status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
    status.textContent = "영상 파일 경로 연결 필요"; panel.append(status);
    const resize = () => {
        if (!node.computeSize || !node.setSize || node._sharedRemoved) return;
        const width = node.size?.[0] || 320;
        const height = node.computeSize()[1];
        if (height !== node.size?.[1]) node.setSize([width, height]);
    };
    const previous = node.onExecuted;
    node.onExecuted = function (message) {
        previous?.apply(this, arguments);
        if (node._sharedRemoved) return;
        const result = (Array.isArray(message?.shared_media) ? message.shared_media : []).find((item) => item?.final === true && /^\/remote-vae\/media\/[a-f0-9]{32}$/.test(item.url));
        if (!result) return;
        releaseVideo(panel.querySelector("video"));
        const video = document.createElement("video"); video.controls = true; video.preload = "metadata";
        video.src = api.apiURL(result.url); video.setAttribute("aria-label", "완성 영상");
        video.addEventListener("loadedmetadata", resize);
        const link = document.createElement("a"); link.href = video.src; link.textContent = "영상 열기";
        link.target = "_blank"; link.rel = "noopener";
        const details = document.createElement("details"); const summary = document.createElement("summary"); summary.textContent = "상세";
        const cleanup = message.source_cleanup?.[0] === true;
        const diagnostic = document.createElement("p"); diagnostic.textContent = cleanup ? "중간 파일 정리됨"
            : message.source_cleanup?.[0] === false ? "중간 파일 보존" : "중간 파일 정리 상태 확인 필요";
        details.append(summary, diagnostic); details.addEventListener("toggle", resize);
        video.addEventListener("error", () => {
            if (node._sharedRemoved || panel.querySelector("video") !== video) return;
            status.textContent = "미리보기 로드 실패"; link.textContent = "영상 다시 열기";
            resize();
        });
        status.textContent = "영상 완료";
        panel.replaceChildren(status, video, link, details);
        if (cleanup) document.dispatchEvent(new CustomEvent("remote-vae:finalized", {detail: result}));
        resize();
    };
    const widget = node.addDOMWidget("shared_video_result", "custom", panel, {serialize: false});
    widget.serialize = false;
    widget.computeSize = (width) => [width, Math.max(60, Math.ceil(Math.max(panel.scrollHeight || 0, panel.offsetHeight || 0)))];
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
    observer?.observe(panel);
    const removed = node.onRemoved;
    node.onRemoved = function () {
        node._sharedRemoved = true; releaseVideo(panel.querySelector("video")); observer?.disconnect();
        return removed?.apply(this, arguments);
    };
    resize();
}

if (typeof document !== "undefined" && !document.getElementById("remote-vae-delivery-style")) {
    const link = document.createElement("link"); link.id = "remote-vae-delivery-style"; link.rel = "stylesheet";
    link.href = new URL("./vae_delivery.css", import.meta.url).href; document.head.append(link);
}
app.registerExtension({
    name: "RemoteWorker.SharedVideo",
    nodeCreated(node) { setupResult(node); },
    loadedGraphNode(node) { setupResult(node); },
});

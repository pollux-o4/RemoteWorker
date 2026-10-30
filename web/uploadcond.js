import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { readError } from "./remote_choices.js";

export function uploadCond(file, node) {
    const run = async () => {
        const widget = node.widgets?.find(item => item.name === "filename");
        const button = node._conditioningUpload;
        if (!widget) return false;
        const label = (text, detail = "") => {
            if (button) {
                button.label = text;
                button.options = { ...(button.options || {}), tooltip: detail || "파일 선택 · .cond 드롭·붙여넣기 가능" };
            }
            app.graph?.setDirtyCanvas?.(true, true);
        };
        label("업로드 중…", file.name);
        try {
            if (!file.name.toLowerCase().endsWith(".cond")) throw new Error(".cond 파일을 선택하세요.");
            const body = new FormData();
            body.append("image", file, file.name);
            body.append("overwrite", "false");
            const response = await api.fetchApi("/upload/image", { method: "POST", body });
            if (!response.ok) throw new Error(await readError(response));
            const data = await response.json();
            if (typeof data.name !== "string" || !data.name || (data.subfolder != null && typeof data.subfolder !== "string")) {
                throw new Error("업로드 결과를 확인할 수 없습니다. 파일 목록을 확인하세요.");
            }
            const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
            const values = Array.isArray(widget.options?.values) ? widget.options.values : [];
            widget.options = { ...(widget.options || {}), values: [...new Set([...values, name])] };
            widget.value = name;
            try {
                widget.callback?.(name);
            } catch (error) {
                const detail = error instanceof Error ? error.message : String(error);
                label("업로드 완료 · 선택 확인", `${name}\n${detail}`);
                app.extensionManager?.toast?.add({ severity: "warn", summary: ".cond 저장 완료",
                    detail: "파일 저장 완료. 파일 목록에서 선택을 확인하세요.", life: 7000 });
                console.warn("[LoadConditioning] selection update failed", error);
                return true;
            }
            label(".cond 업로드", name);
            return true;
        } catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            label("업로드 실패 · 다시 선택", detail);
            app.extensionManager?.toast?.add({ severity: "error", summary: ".cond 업로드 실패", detail: `${detail.slice(0, 180)} 기존 선택 유지. 파일을 다시 선택하세요.`, life: 7000 });
            console.warn("[LoadConditioning] upload failed", error);
            return false;
        }
    };
    node._conditioningUploadQueue = (node._conditioningUploadQueue || Promise.resolve()).then(run, run);
    return node._conditioningUploadQueue;
}

export function pickAndUpload(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".cond";
    input.multiple = true;
    input.style.display = "none";
    document.body.append(input);
    input.oncancel = () => input.remove();
    input.onchange = async () => {
        try {
            for (const file of input.files || []) await uploadCond(file, node);
        } finally {
            input.remove();
        }
    };
    input.click();
}

app.registerExtension({
    name: "ConditioningIO.Upload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "LoadConditioning") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const button = node.addWidget("button", "conditioning_upload", null, () => pickAndUpload(node), { serialize: false });
            button.label = ".cond 업로드";
            button.serialize = false;
            button.options = { ...(button.options || {}), tooltip: "파일 선택 · .cond 드롭·붙여넣기 가능" };
            node._conditioningUpload = button;
            node.onDragOver = event => !!event?.dataTransfer?.types?.includes?.("Files");
            node.onDragDrop = event => {
                const files = [...(event.dataTransfer?.files || [])].filter(file => file.name.toLowerCase().endsWith(".cond"));
                if (!files.length) return false;
                for (const file of files) uploadCond(file, node);
                return true;
            };
            return result;
        };
    },
    setup() {
        window.addEventListener("paste", async event => {
            const files = [...(event.clipboardData?.files || [])].filter(file => file.name.toLowerCase().endsWith(".cond"));
            if (!files.length) return;
            const node = Object.values(app.canvas?.selected_nodes || {}).find(item => (item.comfyClass || item.type) === "LoadConditioning");
            if (!node) return;
            event.preventDefault();
            for (const file of files) await uploadCond(file, node);
        });
    },
});

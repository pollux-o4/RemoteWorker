// LoadConditioning 노드에 LoadImage 식 업로드 UX 부여: 업로드 버튼 + 드래그-드롭 + 붙여넣기.
// .cond 파일을 /upload/image (임의 바이트 허용) 로 서버 input/ 에 올리고 드롭다운에 잡는다.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function uploadCond(file, node) {
    const w = node.widgets?.find((x) => x.name === "filename");
    if (!w) return;
    try {
        const body = new FormData();
        body.append("image", file, file.name);   // /upload/image 는 필드명 'image' 를 받음(내용 검증 없음)
        body.append("overwrite", "true");
        const resp = await api.fetchApi("/upload/image", { method: "POST", body });
        if (resp.status !== 200) { console.error("[LoadConditioning] upload 실패", resp.status); return; }
        const data = await resp.json();
        const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
        if (!w.options.values.includes(name)) w.options.values.push(name);
        w.value = name;
        w.callback?.(name);
        app.graph.setDirtyCanvas(true, true);
        console.log("[LoadConditioning] 업로드됨 →", name);
    } catch (e) {
        console.error("[LoadConditioning] upload 예외", e);
    }
}

function pickAndUpload(node) {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = ".cond";
    inp.multiple = true;
    inp.style.display = "none";
    document.body.append(inp);
    inp.onchange = async () => {
        for (const f of inp.files) await uploadCond(f, node);
        inp.remove();
    };
    inp.click();
}

app.registerExtension({
    name: "ConditioningIO.Upload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "LoadConditioning") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            node.addWidget("button", "📂 .cond 업로드 (또는 드래그/붙여넣기)", "", () => pickAndUpload(node));

            // 노드 위로 파일 드래그-드롭
            node.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
            node.onDragDrop = (e) => {
                const files = [...(e.dataTransfer?.files || [])].filter((f) => f.name.toLowerCase().endsWith(".cond"));
                if (!files.length) return false;
                (async () => { for (const f of files) await uploadCond(f, node); })();
                return true;
            };
            return r;
        };
    },

    // 탐색기에서 파일 복사(Ctrl+C) 후 캔버스에서 붙여넣기(Ctrl+V) — 선택된 LoadConditioning 노드로 업로드.
    setup() {
        window.addEventListener("paste", async (e) => {
            const files = [...(e.clipboardData?.files || [])].filter((f) => f.name.toLowerCase().endsWith(".cond"));
            if (!files.length) return;
            const sel = Object.values(app.canvas?.selected_nodes || {});
            const node = sel.find((n) => n.comfyClass === "LoadConditioning");
            if (!node) return;
            e.preventDefault();
            for (const f of files) await uploadCond(f, node);
        });
    },
});

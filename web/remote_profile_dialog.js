import { remoteRequestConfig, remoteServerRevision, fetchJson, normalizeRemoteOptions } from "./remote_choices.js";

export async function managePromptProfile(config, payload) {
    try {
        return await fetchJson("/remote-prompt-enhancer/profiles/manage", { ...config, ...payload });
    } catch (error) {
        const sameOrigin = new URL(config.server_url).origin === window.location.origin;
        if (sameOrigin && error instanceof Error && error.message.startsWith("HTTP 405")) {
            return fetchJson("/remote-prompt-enhancer/profiles", payload);
        }
        throw error;
    }
}

export function addProfileField(parent, label, control) {
    const grouped = control.tagName === "DIV";
    const row = document.createElement(grouped ? "fieldset" : "label");
    Object.assign(row.style, { display: "grid", gap: "4px", margin: "0 0 12px", border: "0", padding: "0", minWidth: "0" });
    const title = document.createElement(grouped ? "legend" : "span");
    title.textContent = label;
    row.append(title, control);
    parent.append(row);
    return row;
}

export function selectProfileValue(select, value) {
    if (value && ![...select.options].some((option) => option.value === value)) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = `${value} (서버 목록에 없음)`;
        select.append(option);
    }
    select.value = value;
}

export function makeSelect(values, value) {
    const select = document.createElement("select");
    for (const item of values) {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        select.append(option);
    }
    selectProfileValue(select, value);
    return select;
}

const errorText = (error) => error instanceof Error ? error.message : String(error);
const newKey = "@new";
const profileFields = ["id", "description", "clip_name", "clip_type", "device", "max_length", "system_prompt", "accepted_media"];
export function profileMutationObserved(profiles, mutation) {
    const target = mutation.action === "delete" ? mutation.profile_id : mutation.profile.id;
    const actual = profiles.find((profile) => profile.id === target);
    if (mutation.action === "delete") return !actual;
    return Boolean(actual && profileFields.every((field) => {
        const expected = mutation.profile[field];
        return field === "accepted_media"
            ? actual[field]?.length === expected.length && expected.every((media) => actual[field].includes(media))
            : actual[field] === (typeof expected === "string" ? expected.trim() : expected);
    }));
}

export async function openPromptProfileManager(node, adapter, options = {}) {
    if (node._remotePromptManagerDialog?.open) {
        node._remotePromptManagerDialog.focus();
        return node._remotePromptManagerDialog;
    }
    const opener = document.activeElement;
    const dialog = document.createElement("dialog");
    node._remotePromptManagerDialog = dialog;
    dialog.setAttribute("aria-label", "프로필 관리");
    Object.assign(dialog.style, {
        boxSizing: "border-box", width: "min(600px, calc(100vw - 24px))", maxHeight: "calc(100dvh - 24px)",
        overflow: "auto", padding: "20px", borderRadius: "8px", overflowWrap: "anywhere",
        border: "1px solid var(--border-color, #555)", background: "var(--comfy-menu-bg, #252525)",
        color: "var(--input-text, #eee)", font: "inherit",
    });
    let config;
    let serverRevision;
    let profiles = [];
    let clips = null;
    let loaded = false;
    let busy = false;
    let stale = false;
    let sequence = 0;
    let mode = "view";
    let pending = null;
    let activeKey = node.widgets?.find((widget) => widget.name === "profile_id")?.value || "";
    let drafts = new Map();
    const sessions = node._remotePromptProfileDrafts instanceof Map ? node._remotePromptProfileDrafts : new Map();
    node._remotePromptProfileDrafts = sessions;
    const make = (tag, text, role) => {
        const element = document.createElement(tag);
        if (text) element.textContent = text;
        if (role) element.dataset.profileRole = role;
        return element;
    };
    const button = (text, role) => {
        const control = make("button", text, role);
        control.type = "button";
        Object.assign(control.style, { minHeight: "36px", padding: "6px 12px", font: "inherit", cursor: "pointer" });
        return control;
    };
    const heading = make("h3", "프로필");
    heading.style.margin = "0 0 4px";
    const origin = make("small", "", "server");
    const status = make("p", "조회 중…", "status");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    const errorDetails = make("details");
    const errorBody = make("pre", "", "error");
    Object.assign(errorBody.style, { whiteSpace: "pre-wrap", maxHeight: "8rem", overflow: "auto", margin: "8px 0" });
    errorDetails.append(make("summary", "오류 상세"), errorBody);
    errorDetails.hidden = true;
    const retry = button("다시 조회", "retry");
    retry.hidden = true;
    const pickerRow = make("div");
    Object.assign(pickerRow.style, { display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "end" });
    const picker = makeSelect([], "");
    picker.dataset.profileRole = "picker";
    picker.setAttribute("aria-label", "프로필 선택");
    Object.assign(picker.style, { flex: "1 1 180px", minWidth: "0" });
    const add = button("추가", "add");
    pickerRow.append(picker, add);
    const summary = make("section");
    const descriptionText = make("p", "", "description");
    Object.assign(descriptionText.style, { whiteSpace: "pre-wrap", maxHeight: "8rem", overflow: "auto" });
    const settings = make("details");
    settings.open = Boolean(options.details);
    const settingsBody = make("pre");
    Object.assign(settingsBody.style, { whiteSpace: "pre-wrap", maxHeight: "12rem", overflow: "auto", font: "inherit" });
    settings.append(make("summary", "설정"), settingsBody);
    const edit = button("편집", "edit");
    const remove = button("삭제", "delete");
    const readActions = make("div");
    Object.assign(readActions.style, { display: "flex", gap: "8px", margin: "12px 0" });
    readActions.append(edit, remove);
    summary.append(descriptionText, settings, readActions);
    const form = make("form");
    const id = make("input", "", "id");
    id.required = true;
    id.pattern = "[a-z0-9][a-z0-9-]{0,63}";
    id.maxLength = 64;
    id.title = "영문 소문자·숫자·하이픈, 최대 64자";
    const description = make("textarea", "", "description-input");
    description.rows = 2;
    description.required = true;
    const systemPrompt = make("textarea", "", "system-prompt");
    systemPrompt.rows = 4;
    systemPrompt.required = true;
    const mediaGroup = make("div");
    Object.assign(mediaGroup.style, { display: "flex", flexWrap: "wrap", gap: "16px" });
    const media = new Map();
    for (const [kind, labelText] of [["image", "이미지"], ["video", "영상"]]) {
        const label = make("label");
        const checkbox = make("input", "", kind);
        checkbox.type = "checkbox";
        label.append(checkbox, document.createTextNode(` ${labelText}`));
        mediaGroup.append(label);
        media.set(kind, checkbox);
    }
    const clipName = makeSelect([], "");
    const clipType = makeSelect([], "");
    const device = makeSelect([], "");
    clipName.required = clipType.required = device.required = true;
    clipName.dataset.profileRole = "clip-name";
    const maxLength = make("input", "", "max-length");
    Object.assign(maxLength, { type: "number", min: "16", max: "2048", step: "1", required: true });
    const advanced = make("details");
    advanced.append(make("summary", "생성 설정"));
    addProfileField(advanced, "CLIP 타입", clipType);
    addProfileField(advanced, "장치", device);
    addProfileField(advanced, "최대 토큰", maxLength);
    addProfileField(form, "ID", id);
    addProfileField(form, "설명", description);
    addProfileField(form, "시스템 지침", systemPrompt);
    addProfileField(form, "참조 미디어", mediaGroup);
    addProfileField(form, "모델", clipName);
    form.append(advanced);
    const confirmation = make("section");
    const confirmText = make("p", "", "delete-target");
    const back = button("돌아가기", "delete-cancel");
    const confirmDelete = button("삭제", "delete-confirm");
    confirmation.append(confirmText, back, confirmDelete);
    const footer = make("div");
    Object.assign(footer.style, {
        display: "flex", flexWrap: "wrap", gap: "8px", justifyContent: "flex-end", position: "sticky",
        bottom: "-20px", padding: "12px 0", background: "var(--comfy-menu-bg, #252525)",
    });
    const reset = button("되돌리기", "reset");
    const save = button("저장", "save");
    const close = button("닫기", "close");
    footer.append(reset, close, save);
    dialog.append(heading, origin, status, errorDetails, retry, pickerRow, summary, form, confirmation, footer);
    for (const control of dialog.querySelectorAll("input:not([type=checkbox]), select, textarea")) {
        Object.assign(control.style, { width: "100%", minWidth: "0", boxSizing: "border-box", minHeight: "36px", padding: "6px 8px", font: "inherit", background: "var(--comfy-input-bg, #333)", color: "inherit", border: "1px solid var(--border-color, #555)", borderRadius: "4px" });
    }
    const fullProfiles = (payload) => {
        const values = adapter.normalize(payload);
        if (values.some((profile) => ["clip_name", "clip_type", "device", "system_prompt"].some((key) => typeof profile[key] !== "string" || !profile[key].trim()) || !Number.isInteger(profile.max_length))) {
            throw new Error("프로필 설정 응답이 불완전합니다. 다시 조회하세요.");
        }
        return values;
    };
    const selected = () => profiles.find((profile) => profile.id === activeKey);
    const serverMatches = () => config && remoteServerRevision(node) === serverRevision && node.widgets?.find((widget) => widget.name === "server_url")?.value === config.server_url;
    const notify = (text, error = null) => {
        status.textContent = text;
        status.hidden = !text;
        errorDetails.hidden = !error;
        errorBody.textContent = error ? errorText(error) : "";
    };
    const readForm = () => ({
        id: id.value, description: description.value, system_prompt: systemPrompt.value,
        clip_name: clipName.value, clip_type: clipType.value, device: device.value,
        max_length: Number(maxLength.value), accepted_media: [...media].filter(([, control]) => control.checked).map(([kind]) => kind),
    });
    const stash = () => { if (mode === "edit" || mode === "new") drafts.set(activeKey, readForm()); };
    const fill = (profile) => {
        const value = profile || { id: "", description: "", system_prompt: "", clip_name: clips?.clip_name[0] || "", clip_type: clips?.clip_type[0] || "", device: clips?.device[0] || "", max_length: 256, accepted_media: ["image"] };
        id.value = value.id;
        description.value = value.description;
        systemPrompt.value = value.system_prompt;
        for (const [select, name] of [[clipName, "clip_name"], [clipType, "clip_type"], [device, "device"]]) {
            select.replaceChildren(...makeSelect(clips?.[name] || (value[name] ? [value[name]] : []), value[name]).options);
            selectProfileValue(select, value[name]);
        }
        maxLength.value = String(value.max_length);
        for (const [kind, control] of media) control.checked = value.accepted_media.includes(kind);
    };
    const populatePicker = () => {
        const ids = profiles.map((profile) => profile.id);
        if (activeKey === newKey) ids.unshift(newKey);
        else if (!ids.includes(activeKey)) ids.unshift(activeKey);
        picker.replaceChildren(...makeSelect(ids, activeKey).options);
        for (const option of picker.options) if (option.value === newKey) option.textContent = "새 프로필";
    };
    const render = () => {
        const profile = selected();
        const writing = mode === "edit" || mode === "new";
        picker.disabled = busy || !loaded || Boolean(pending) || mode === "delete";
        add.disabled = busy || Boolean(pending) || !loaded || stale || (!clips && !profiles.length);
        edit.disabled = remove.disabled = busy || Boolean(pending) || !profile || stale;
        summary.hidden = !loaded || writing || mode === "delete";
        form.hidden = !writing;
        confirmation.hidden = mode !== "delete";
        save.hidden = reset.hidden = !writing;
        save.disabled = busy || Boolean(pending) || stale || !loaded;
        reset.disabled = busy || Boolean(pending) || stale;
        close.disabled = busy && Boolean(pending);
        back.disabled = confirmDelete.disabled = busy || Boolean(pending) || stale;
        retry.disabled = busy;
        id.readOnly = mode !== "new";
        for (const control of form.querySelectorAll("input, select, textarea")) control.disabled = busy || Boolean(pending) || stale;
        dialog.setAttribute("aria-busy", String(busy));
        descriptionText.textContent = profile?.description || (profiles.length ? "선택한 프로필 없음 · 기존 선택 유지" : "프로필 없음 · 추가");
        settings.hidden = !profile;
        settingsBody.textContent = profile ? `${profile.clip_name}\n${profile.clip_type} · ${profile.device} · ${profile.max_length} 토큰\n${profile.accepted_media.map((kind) => kind === "image" ? "이미지" : "영상").join(" · ")}\n\n${profile.system_prompt}` : "";
        confirmText.textContent = `“${activeKey}” 프로필을 삭제할까요?`;
    };
    const markStale = () => {
        stale = true;
        notify(pending ? "이전 서버 결과 확인 필요 · 입력 유지" : "서버 변경됨 · 다시 열기");
        retry.textContent = "현재 서버 열기";
        retry.hidden = false;
        render();
    };
    const applyConfirmed = (payload) => {
        if (!serverMatches()) { markStale(); return false; }
        const fresh = fullProfiles(payload);
        const widget = node.widgets?.find((item) => item.name === "profile_id");
        const target = pending.action === "delete" ? pending.profile_id : pending.profile.id;
        adapter.apply(node, payload);
        if (widget && pending.action !== "delete" && fresh.some((profile) => profile.id === target)) widget.value = target;
        if (widget && pending.action === "delete" && widget.value === target) widget.value = "";
        widget?.callback?.(widget.value);
        drafts.delete(pending.action === "delete" ? target : activeKey);
        activeKey = pending.action === "delete" ? (widget?.value || "") : target;
        mode = "view";
        pending = null;
        dialog.close();
        return true;
    };
    const reconcile = async () => {
        if (!serverMatches()) { markStale(); return; }
        busy = true; notify("결과 확인 중…"); render();
        try {
            const payload = await fetchJson("/remote-prompt-enhancer/profile-options", config);
            if (!dialog.open) return;
            if (!serverMatches()) { markStale(); return; }
            const fresh = fullProfiles(payload);
            if (pending.confirmed || profileMutationObserved(fresh, pending)) { applyConfirmed(payload); return; }
            profiles = fresh; loaded = true; pending = null; populatePicker();
            if (drafts.has(activeKey)) { mode = activeKey === newKey ? "new" : "edit"; fill(drafts.get(activeKey)); }
            notify("저장 확인 안 됨 · 입력 유지", new Error("현재 목록을 확인했습니다. 입력을 검토한 뒤 다시 저장하세요."));
            retry.hidden = true;
        } catch (error) {
            if (!dialog.open) return;
            if (!serverMatches()) { markStale(); return; }
            notify(pending?.confirmed ? "저장됨 · 목록 다시 조회" : "저장 여부 미확인 · 결과 확인", error);
            retry.textContent = "결과 확인"; retry.hidden = false;
        } finally { busy = false; render(); if (dialog.open && !retry.hidden) retry.focus(); }
    };
    const load = async () => {
        const ticket = ++sequence;
        busy = true; notify("조회 중…"); retry.hidden = true; render();
        const [profileResult, clipResult] = await Promise.allSettled([
            fetchJson("/remote-prompt-enhancer/profile-options", config),
            fetchJson("/remote-conditioning/clip-loader-options", config),
        ]);
        if (!dialog.open || ticket !== sequence) return;
        busy = false;
        if (!serverMatches()) { markStale(); return; }
        try {
            if (profileResult.status !== "fulfilled") throw profileResult.reason;
            profiles = fullProfiles(profileResult.value);
            loaded = true;
            if (!activeKey && profiles.length) activeKey = profiles[0].id;
            let clipError = clipResult.status === "rejected" ? clipResult.reason : null;
            if (!clipError) {
                try { clips = normalizeRemoteOptions(clipResult.value); }
                catch (error) { clipError = error; }
            }
            populatePicker();
            if (drafts.has(activeKey)) { mode = activeKey === newKey ? "new" : "edit"; fill(drafts.get(activeKey)); }
            notify(clipError ? "모델 목록 조회 실패 · 설정 유지" : (drafts.has(activeKey) ? "입력 복원됨" : ""), clipError);
            retry.hidden = !clipError;
            retry.textContent = "다시 조회";
        } catch (error) {
            notify("조회 실패 · 다시 조회", error); retry.hidden = false;
        }
        render();
        if (!retry.hidden) retry.focus();
        else if (document.activeElement === dialog || document.activeElement === close) picker.focus();
    };
    const mutate = async (payload) => {
        if (busy || pending) return;
        if (!serverMatches()) { markStale(); return; }
        stash(); pending = { ...payload, confirmed: false };
        busy = true; notify(payload.action === "delete" ? "삭제 중…" : "저장 중…"); render();
        try {
            await managePromptProfile(config, payload);
            pending.confirmed = true;
            if (!dialog.open) return;
            if (!serverMatches()) { markStale(); return; }
            await reconcile();
        } catch (error) {
            if (!dialog.open) return;
            if (!serverMatches()) { markStale(); return; }
            notify("저장 여부 미확인 · 결과 확인", error);
            retry.textContent = "결과 확인"; retry.hidden = false;
        } finally { busy = false; render(); if (dialog.open && !retry.hidden) retry.focus(); }
    };
    picker.addEventListener("change", () => {
        stash(); activeKey = picker.value; mode = "view"; notify(""); render();
    });
    add.addEventListener("click", () => {
        stash(); activeKey = newKey; mode = "new"; fill(drafts.get(newKey)); populatePicker(); notify(""); render(); id.focus();
    });
    edit.addEventListener("click", () => {
        mode = "edit"; fill(drafts.get(activeKey) || selected()); notify(""); render(); description.focus();
    });
    remove.addEventListener("click", () => { mode = "delete"; notify(""); render(); back.focus(); });
    back.addEventListener("click", () => { mode = "view"; render(); remove.focus(); });
    confirmDelete.addEventListener("click", () => mutate({ action: "delete", profile_id: activeKey }));
    reset.addEventListener("click", () => { drafts.delete(activeKey); fill(mode === "new" ? null : selected()); notify(""); id.focus(); });
    const submit = (event) => {
        event?.preventDefault();
        if (busy || pending || stale) return;
        const value = readForm();
        id.setCustomValidity(mode === "new" && profiles.some((profile) => profile.id === value.id) ? "이미 있는 ID · 기존 프로필 편집" : "");
        description.setCustomValidity(!value.description.trim() ? "설명 입력 필요" : [...value.description.trim()].length > 480 ? "설명은 최대 480자" : "");
        systemPrompt.setCustomValidity(value.system_prompt.trim() ? "" : "시스템 지침 입력 필요");
        media.get("image").setCustomValidity(value.accepted_media.length ? "" : "참조 미디어 선택 필요");
        media.get("video").setCustomValidity(value.accepted_media.includes("video") && value.clip_type !== "minimax" ? "영상은 minimax 타입에서 사용" : "");
        if (!maxLength.checkValidity() || media.get("video").validationMessage) advanced.open = true;
        if (!form.reportValidity()) return;
        mutate({ action: "upsert", profile: value });
    };
    form.addEventListener("submit", submit);
    save.addEventListener("click", submit);
    retry.addEventListener("click", () => {
        if (stale) { dialog.close(); queueMicrotask(() => openPromptProfileManager(node, adapter, options)); }
        else if (pending) reconcile();
        else { stash(); load(); }
    });
    close.addEventListener("click", () => dialog.close());
    dialog.addEventListener("cancel", (event) => {
        if (busy && pending) event.preventDefault();
        else if (mode === "delete") { event.preventDefault(); mode = "view"; render(); remove.focus(); }
    });
    dialog.addEventListener("close", () => {
        stash(); sequence++;
        if (config) sessions.set(config.server_url, { activeKey, drafts, pending });
        dialog.remove();
        if (node._remotePromptManagerDialog === dialog) node._remotePromptManagerDialog = null;
        opener?.focus?.();
    }, { once: true });
    document.body.append(dialog); render(); dialog.showModal();
    try {
        config = remoteRequestConfig(node);
        serverRevision = remoteServerRevision(node);
        origin.textContent = new URL(config.server_url).host;
        origin.title = config.server_url;
        const saved = sessions.get(config.server_url);
        if (saved) {
            drafts = saved.drafts; activeKey = saved.activeKey; pending = saved.pending || null;
            if (drafts.has(activeKey)) { mode = activeKey === newKey ? "new" : "edit"; fill(drafts.get(activeKey)); }
        }
        if (pending) await reconcile();
        else await load();
    } catch (error) { busy = false; notify("서버 설정 확인", error); render(); }
    return dialog;
}

import { remoteRequestConfig, fetchJson, normalizeRemoteOptions } from "./remote_choices.js";


export async function managePromptProfile(config, payload) {
    try {
        return await fetchJson("/remote-prompt-enhancer/profiles/manage", { ...config, ...payload });
    } catch (error) {
        // A page served by peer can use its native registry route immediately.
        // This preserves profile management during a frontend refresh when an
        // older long-lived server has not yet registered the caller proxy route.
        const sameNviOrigin = new URL(config.server_url).origin === window.location.origin;
        if (sameNviOrigin && error instanceof Error && error.message.startsWith("HTTP 405")) {
            return fetchJson("/remote-prompt-enhancer/profiles", payload);
        }
        throw error;
    }
}

export function addProfileField(dialog, label, control) {
    const row = document.createElement(control.tagName === "DIV" ? "fieldset" : "label");
    row.style.display = "grid";
    row.style.gap = "4px";
    row.style.marginBottom = "16px";
    row.style.border = "0";
    row.style.padding = "0";
    row.style.minWidth = "0";
    const title = document.createElement(control.tagName === "DIV" ? "legend" : "span");
    title.textContent = label;
    row.append(title, control);
    dialog.append(row);
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

export async function openPromptProfileManager(node, profilesAdapter) {
    try {
        const config = remoteRequestConfig(node);
        const [profilePayload, clipPayload] = await Promise.all([
            fetchJson("/remote-prompt-enhancer/profile-options", config),
            fetchJson("/remote-conditioning/clip-loader-options", config),
        ]);
        const profiles = profilesAdapter.normalize(profilePayload);
        const clips = normalizeRemoteOptions(clipPayload);
        const currentId = node.widgets?.find((widget) => widget.name === "profile_id")?.value || "";
        const current = profiles.find((profile) => profile.id === currentId) || profiles[0];
        const dialog = document.createElement("dialog");
        const opener = document.activeElement;
        Object.assign(dialog.style, {
            width: "min(640px, calc(100vw - 40px))", maxHeight: "calc(100vh - 40px)",
            overflowY: "auto", padding: "24px", borderRadius: "12px",
            border: "1px solid var(--border-color, #555)",
            background: "var(--comfy-menu-bg, #252525)", color: "var(--input-text, #eee)",
        });
        dialog.setAttribute("aria-label", "Remote Prompt Enhancer 프로필 관리");
        const heading = document.createElement("h3");
        heading.textContent = "Remote Prompt Enhancer 프로필 관리";
        dialog.append(heading);

        const action = makeSelect(["create", "update", "delete"], "update");
        for (const option of action.options) {
            option.textContent = { create: "생성", update: "수정", delete: "삭제" }[option.value];
        }
        const newId = document.createElement("input");
        newId.value = "";
        newId.pattern = "[a-z0-9][a-z0-9-]{0,63}";
        const existingId = makeSelect(profiles.map((profile) => profile.id), current?.id || profiles[0]?.id);
        const description = document.createElement("textarea");
        description.rows = 3;
        const systemPrompt = document.createElement("textarea");
        systemPrompt.rows = 5;
        const acceptedMedia = document.createElement("div");
        const mediaCheckboxes = new Map();
        for (const media of ["image", "video"]) {
            const label = document.createElement("label");
            label.style.marginRight = "12px";
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.value = media;
            label.append(checkbox, ` ${media === "image" ? "이미지" : "영상"}`);
            acceptedMedia.append(label);
            mediaCheckboxes.set(media, checkbox);
        }
        const clipName = makeSelect(clips.clip_name, current?.clip_name || clips.clip_name[0]);
        const clipType = makeSelect(clips.clip_type, current?.clip_type || clips.clip_type[0]);
        const device = makeSelect(clips.device, current?.device || clips.device[0]);
        const maxLength = document.createElement("input");
        maxLength.type = "number";
        maxLength.min = "16";
        maxLength.max = "2048";
        maxLength.value = String(current?.max_length || 256);
        const editable = [description, systemPrompt, acceptedMedia, clipName, clipType, device, maxLength];
        const syncProfile = () => {
            const creating = action.value === "create";
            newId.hidden = !creating;
            existingId.hidden = creating;
            const profile = profiles.find((item) => item.id === existingId.value);
            if (!creating && profile) {
                description.value = profile.description;
                selectProfileValue(clipName, profile.clip_name);
                selectProfileValue(clipType, profile.clip_type);
                selectProfileValue(device, profile.device);
                maxLength.value = String(profile.max_length);
                systemPrompt.value = profile.system_prompt;
                for (const [media, checkbox] of mediaCheckboxes) checkbox.checked = profile.accepted_media.includes(media);
            }
            const deleting = action.value === "delete";
            editable.forEach((control) => { control.disabled = deleting; });
            save.textContent = { create: "생성", update: "수정", delete: "삭제" }[action.value];
        };

        addProfileField(dialog, "작업", action);
        const idField = document.createElement("div");
        newId.setAttribute("aria-label", "새 프로필 ID");
        existingId.setAttribute("aria-label", "기존 프로필 ID");
        idField.append(newId, existingId);
        addProfileField(dialog, "프로필 ID", idField);
        addProfileField(dialog, "설명", description);
        addProfileField(dialog, "시스템 프롬프트", systemPrompt);
        addProfileField(dialog, "참조 미디어", acceptedMedia);
        addProfileField(dialog, "CLIP 모델", clipName);
        addProfileField(dialog, "CLIP 타입", clipType);
        addProfileField(dialog, "장치", device);
        addProfileField(dialog, "최대 토큰", maxLength);
        const buttons = document.createElement("div");
        buttons.style.display = "flex";
        buttons.style.gap = "8px";
        const save = document.createElement("button");
        const cancel = document.createElement("button");
        cancel.textContent = "취소";
        buttons.append(save, cancel);
        const status = document.createElement("p");
        status.setAttribute("role", "alert");
        status.hidden = true;
        status.style.color = "var(--error-text, #ffb4ab)";
        dialog.append(status, buttons);
        for (const control of dialog.querySelectorAll("input:not([type=checkbox]), select, textarea")) {
            Object.assign(control.style, {
                width: "100%", boxSizing: "border-box", minHeight: "36px", padding: "8px 10px",
                font: "inherit", fontSize: "14px", lineHeight: "1.5", borderRadius: "6px",
                border: "1px solid var(--border-color, #555)",
                background: "var(--comfy-input-bg, #333)", color: "inherit",
            });
        }
        Object.assign(buttons.style, {
            position: "sticky", bottom: "-24px", padding: "16px 0", marginBottom: "-24px",
            background: "var(--comfy-menu-bg, #252525)", justifyContent: "flex-end",
            borderTop: "1px solid var(--border-color, #555)",
        });
        for (const button of [save, cancel]) {
            Object.assign(button.style, { minHeight: "36px", minWidth: "76px", padding: "8px 16px", borderRadius: "6px", font: "inherit" });
        }
        existingId.addEventListener("change", syncProfile);
        action.addEventListener("change", () => {
            if (action.value === "create") {
                newId.value = "";
                description.value = "";
                systemPrompt.value = current?.system_prompt || "";
                for (const [media, checkbox] of mediaCheckboxes) checkbox.checked = media === "image";
            }
            syncProfile();
        });
        syncProfile();
        cancel.addEventListener("click", () => dialog.close());
        save.addEventListener("click", async () => {
            if (save.disabled) return;
            status.hidden = true;
            const controls = [...dialog.querySelectorAll("input, select, textarea, button")];
            const disabled = controls.map((control) => control.disabled);
            controls.forEach((control) => { control.disabled = true; });
            dialog.setAttribute("aria-busy", "true");
            try {
                const profileId = action.value === "create" ? newId.value : existingId.value;
                const existing = profiles.some((profile) => profile.id === profileId);
                if (action.value === "create" && existing) throw new Error("이미 존재하는 프로필입니다. 수정 작업을 선택하세요.");
                if (action.value === "update" && !existing) throw new Error("수정할 기존 프로필을 선택하세요.");
                const payload = { action: action.value === "delete" ? "delete" : "upsert" };
                if (action.value === "delete") {
                    payload.profile_id = profileId;
                } else {
                    payload.profile = {
                        id: profileId,
                        description: description.value,
                        clip_name: clipName.value,
                        clip_type: clipType.value,
                        device: device.value,
                        max_length: Number(maxLength.value),
                        system_prompt: systemPrompt.value,
                        accepted_media: [...mediaCheckboxes].filter(([, checkbox]) => checkbox.checked).map(([media]) => media),
                    };
                }
                await managePromptProfile(config, payload);
                const refreshed = await fetchJson("/remote-prompt-enhancer/profile-options", config);
                profilesAdapter.apply(node, refreshed);
                dialog.close();
            } catch (error) {
                status.textContent = `저장 실패: ${error instanceof Error ? error.message : String(error)}`;
                status.hidden = false;
            } finally {
                controls.forEach((control, index) => { control.disabled = disabled[index]; });
                dialog.removeAttribute("aria-busy");
            }
        });
        dialog.addEventListener("close", () => {
            dialog.remove();
            opener?.focus?.();
        }, { once: true });
        document.body.append(dialog);
        dialog.showModal();
    } catch (error) {
        window.alert(`프로필 관리 화면을 열 수 없습니다: ${error instanceof Error ? error.message : String(error)}`);
    }
}

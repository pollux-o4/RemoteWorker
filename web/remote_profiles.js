import { app } from "../../scripts/app.js";
import { syncPromptMediaInputs } from "./remote_sockets.js";
import { refreshRemoteChoiceOptions } from "./remote_choices.js";
import { openPromptProfileManager } from "./remote_profile_dialog.js";
export const PROFILE_MANAGER_OPTION = "프로필 관리…";
const managerOptions = new Set([PROFILE_MANAGER_OPTION, "Manage profiles..."]);


export function normalizePromptProfiles(payload) {
    const profiles = payload?.profiles;
    if (!Array.isArray(profiles)) throw new Error("원격 프로필 목록이 없습니다");
    const ids = new Set();
    return profiles.map((profile) => {
        if (!profile || typeof profile !== "object" || Array.isArray(profile)) throw new Error("원격 프로필 형식이 올바르지 않습니다");
        const { id, description, accepted_media: acceptedMedia } = profile;
        if (typeof id !== "string" || !/^[a-z0-9][a-z0-9-]{0,63}$/.test(id) || ids.has(id)) {
            throw new Error("원격 프로필 id가 올바르지 않습니다");
        }
        if (typeof description !== "string" || !description.trim()) throw new Error("원격 프로필 설명이 없습니다");
        if (!Array.isArray(acceptedMedia) || !acceptedMedia.length || acceptedMedia.some((item) => item !== "image" && item !== "video")) {
            throw new Error("원격 프로필 미디어 계약이 올바르지 않습니다");
        }
        ids.add(id);
        return { ...profile, id, description: description.trim(), accepted_media: [...new Set(acceptedMedia)] };
    });
}

export function updateProfileDescription(node) {
    const profileWidget = node.widgets?.find((widget) => widget.name === "profile_id");
    const descriptionButton = node._remotePromptDescriptionButton;
    if (!profileWidget || !descriptionButton) return;
    const serverUrl = node.widgets?.find((widget) => widget.name === "server_url")?.value;
    const profile = node._remotePromptProfilesServerUrl === undefined || node._remotePromptProfilesServerUrl === serverUrl
        ? (node._remotePromptProfiles || []).find((item) => item.id === profileWidget.value) : null;
    const description = profile?.description;
    const fullDescription = description || "프로필 선택·설정 확인";
    descriptionButton.name = "프로필 상세";
    descriptionButton.options = { ...(descriptionButton.options || {}), tooltip: fullDescription };
    descriptionButton.callback = () => openPromptProfileManager(node, {
        normalize: normalizePromptProfiles, apply: applyPromptProfiles,
    }, { details: true });
    app.graph?.setDirtyCanvas?.(true, true);
}

export function removeObsoletePromptManagerInput(node) {
    const obsoleteWidgetNames = new Set(["profile_manager", "⚙ 프로필 관리"]);
    for (let index = (node.widgets?.length || 0) - 1; index >= 0; index -= 1) {
        if (!obsoleteWidgetNames.has(node.widgets[index]?.name)) continue;
        if (typeof node.removeWidget === "function") node.removeWidget(index);
        else node.widgets.splice(index, 1);
    }
    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index -= 1) {
        if (node.inputs[index]?.name !== "profile_manager") continue;
        if (typeof node.removeInput === "function") node.removeInput(index);
        else node.inputs.splice(index, 1);
    }
    if (node.properties && Object.hasOwn(node.properties, "profile_manager")) {
        delete node.properties.profile_manager;
    }
}

export function applyPromptProfiles(node, payload) {
    const profiles = normalizePromptProfiles(payload);
    removeObsoletePromptManagerInput(node);
    const widget = node.widgets?.find((item) => item.name === "profile_id");
    if (!widget) throw new Error("profile_id 위젯을 찾을 수 없습니다");
    const ids = profiles.map((profile) => profile.id);
    if (managerOptions.has(widget.value)) widget.value = "";
    if (typeof widget.value === "string" && !ids.includes(widget.value)) ids.unshift(widget.value);
    widget.options = { ...(widget.options || {}), values: [...ids, PROFILE_MANAGER_OPTION] };
    node._remotePromptProfiles = profiles;
    node._remotePromptProfilesServerUrl = node.widgets?.find((item) => item.name === "server_url")?.value;
    setupPromptProfileManagerShortcut(node);
    updateProfileDescription(node);
    // A persisted node may predate the dynamic-media feature.  Refreshing its
    // profile data must materialize the first supported socket too; waiting
    // for a profile-widget change leaves that existing node empty.
    syncPromptMediaInputs(node);
    return profiles.length;
}

export function applyPromptProfilesToVueDefs(definitions, payload) {
    const profiles = normalizePromptProfiles(payload);
    const ids = [...profiles.map((profile) => profile.id), PROFILE_MANAGER_OPTION];
    for (const definition of definitions || []) {
        if (definition?.name !== "RemotePromptEnhancer") continue;
        const profileSpec = definition.input?.required?.profile_id;
        if (!Array.isArray(profileSpec) || !Array.isArray(profileSpec[0])) continue;
        profileSpec[0].splice(0, profileSpec[0].length, ...ids);
        if (profileSpec[1] && typeof profileSpec[1] === "object") {
            profileSpec[1].tooltip = "프로필 선택 · 상세에서 관리";
        }
    }
    return profiles.length;
}

export async function refreshPromptProfiles(node, button) {
    return refreshRemoteChoiceOptions(node, button, {
        loadingKey: "_remotePromptProfilesLoading",
        path: "/remote-prompt-enhancer/profile-options",
        missingSettingsLabel: "서버 설정 필요",
        loadingLabel: "조회 중…",
        serverChangedLabel: "서버 변경됨 · 다시 조회",
        successLabel: (count) => count ? `프로필 ${count}개 · 새로고침` : "프로필 없음 · 다시 조회",
        failureLabel: () => "조회 실패 · 다시 조회",
        warningPrefix: "[Remote Prompt Enhancer] profile lookup failed",
        apply: applyPromptProfiles,
    });
}

export function setupPromptProfileManagerShortcut(node) {
    const widget = node.widgets?.find((item) => item.name === "profile_id");
    if (!widget || widget._remotePromptProfileManagerShortcut) return;
    const previousCallback = widget.callback;
    let selectedProfileId = managerOptions.has(widget.value) ? "" : (widget.value || "");
    widget.callback = function (value, ...args) {
        if (!managerOptions.has(value)) {
            selectedProfileId = value || "";
            return previousCallback?.call(this, value, ...args);
        }
        widget.value = selectedProfileId;
        queueMicrotask(() => openPromptProfileManager(node, { normalize: normalizePromptProfiles, apply: applyPromptProfiles }));
        app.graph?.setDirtyCanvas?.(true, true);
        return undefined;
    };
    widget._remotePromptProfileManagerShortcut = true;
}

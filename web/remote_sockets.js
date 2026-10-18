import { app } from "../../scripts/app.js";
import { setWidgetConfig } from "../../extensions/core/widgetInputs.js";
const LEGACY_VAE_OVERRIDE_INPUTS = new Map([
    ["vae_name_override", "vae_name"],
    ["audio_vae_name_override", "audio_vae_name"],
]);
const VAE_NAME_OVERRIDE_SOCKET = "*";
const IMAGE_INPUT_NAME = /^image_([1-9][0-9]*)$/;
const PROMPT_MEDIA_INPUT_NAME = /^(image|video)_([0-9]+)$/;


export function syncImageInputs(node) {
    const countWidget = node.widgets?.find((widget) => widget.name === "image_count");
    if (!countWidget) throw new Error("image_count 위젯을 찾을 수 없습니다");
    const target = Math.max(0, Math.trunc(Number(countWidget.value) || 0));
    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index -= 1) {
        const match = IMAGE_INPUT_NAME.exec(node.inputs[index]?.name || "");
        if (match && Number(match[1]) > target) node.removeInput(index);
    }
    const existing = new Set(
        (node.inputs || []).map((input) => IMAGE_INPUT_NAME.exec(input.name || ""))
            .filter(Boolean).map((match) => Number(match[1])),
    );
    for (let index = 1; index <= target; index += 1) {
        if (!existing.has(index)) node.addInput(`image_${index}`, "IMAGE", { shape: 7 });
    }
    node.setSize?.(node.computeSize?.() || node.size);
    app.graph?.setDirtyCanvas?.(true, true);
}

export function syncMiniMaxReferenceInputs(node) {
    // Match Comfy's Autogrow UX: one empty trailing socket appears after the
    // last connected reference. No user-managed count widgets are involved.
    const groups = [["ref_image", "IMAGE", 9], ["ref_video", "IMAGE", 3], ["ref_video_audio", "AUDIO", 3], ["ref_audio", "AUDIO", 3]];
    for (const [prefix, type, maximum] of groups) {
        const expression = new RegExp(`^${prefix}_(\\d+)$`);
        const existingInputs = (node.inputs || []).map((input) => ({ input, match: expression.exec(input.name || "") })).filter(({ match }) => match);
        const highestConnected = Math.max(-1, ...existingInputs.filter(({ input }) => input.link != null).map(({ match }) => Number(match[1])));
        const target = Math.min(maximum - 1, Math.max(0, highestConnected + 1));
        for (let index = (node.inputs?.length || 0) - 1; index >= 0; index -= 1) {
            const match = expression.exec(node.inputs[index]?.name || "");
            if (match && Number(match[1]) >= target) node.removeInput(index);
        }
        const existing = new Set((node.inputs || []).map((input) => expression.exec(input.name || "")).filter(Boolean).map((match) => Number(match[1])));
        for (let index = 0; index <= target; index += 1) if (!existing.has(index)) node.addInput(`${prefix}_${index}`, type, { shape: 7 });
    }
    node.setSize?.(node.computeSize?.() || node.size);
    app.graph?.setDirtyCanvas?.(true, true);
}

export function graphLink(node, linkId) {
    const links = node.graph?.links;
    return links instanceof Map ? links.get(linkId) : links?.[linkId];
}

export function targetSlot(node, input) {
    return (node.inputs || []).indexOf(input);
}

export function setTargetSlot(node, input) {
    if (input?.link == null) return;
    const link = graphLink(node, input.link);
    if (link) link.target_slot = targetSlot(node, input);
}

export function nodeInputSpec(node, inputName) {
    return node.constructor?.nodeData?.input?.required?.[inputName] ??
        node.constructor?.nodeData?.input?.optional?.[inputName] ??
        [VAE_NAME_OVERRIDE_SOCKET, {}];
}

export function ensureWidgetBackedVaeOverrideInput(node, legacyName, inputName) {
    const widget = node.widgets?.find((item) => item.name === inputName);
    if (!widget) return null;
    let changed = false;
    let input = node.inputs?.find((item) => item.name === legacyName);
    if (!input) {
        node.addInput(legacyName, VAE_NAME_OVERRIDE_SOCKET, { widget: { name: inputName } });
        input = node.inputs?.[node.inputs.length - 1];
        changed = true;
    }
    if (!input) return null;
    if (input.type !== VAE_NAME_OVERRIDE_SOCKET) {
        input.type = VAE_NAME_OVERRIDE_SOCKET;
        changed = true;
    }
    if (input.widget?.name !== inputName) {
        input.widget = { ...(input.widget || {}), name: inputName };
        changed = true;
    }
    if (input.label !== inputName) {
        input.label = inputName;
        changed = true;
    }
    if (input.localized_name !== inputName) {
        input.localized_name = inputName;
        changed = true;
    }
    setWidgetConfig(input, nodeInputSpec(node, inputName));
    return { input, changed };
}

export function removeInput(node, index) {
    if (typeof node.removeInput === "function") node.removeInput(index);
    else node.inputs.splice(index, 1);
}

export function migratePrimaryVaeWidgetInput(node, inputName, legacyName) {
    let changed = false;
    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index -= 1) {
        const primary = node.inputs[index];
        if (primary?.name !== inputName) continue;
        const target = node.inputs?.find((input) => input.name === legacyName);
        if (primary.link != null && target && target.link == null) {
            target.link = primary.link;
            primary.link = null;
        }
        removeInput(node, index);
        if (target) setTargetSlot(node, target);
        changed = true;
    }
    return changed;
}

export function normalizeMiniMaxVaeInputs(node) {
    let changed = false;
    for (const [legacyName, inputName] of LEGACY_VAE_OVERRIDE_INPUTS) {
        const result = ensureWidgetBackedVaeOverrideInput(node, legacyName, inputName);
        changed = Boolean(result?.changed) || changed;
    }
    for (const [legacyName, inputName] of LEGACY_VAE_OVERRIDE_INPUTS) {
        changed = migratePrimaryVaeWidgetInput(node, inputName, legacyName) || changed;
    }
    if (changed) app.graph?.setDirtyCanvas?.(true, true);
    return changed;
}

export function syncPromptMediaInputs(node) {
    const profileId = node.widgets?.find((widget) => widget.name === "profile_id")?.value;
    const profile = (node._remotePromptProfiles || []).find((item) => item.id === profileId);
    const serverUrl = node.widgets?.find((widget) => widget.name === "server_url")?.value;
    // Unknown or another peer's policy must never remove persisted links.
    if (!profile || (node._remotePromptProfilesServerUrl !== undefined && node._remotePromptProfilesServerUrl !== serverUrl)) return;
    const supported = new Set(profile.accepted_media || []);
    const groups = [["image", 8], ["video", 3]];
    for (const [prefix, maximum] of groups) {
        const inputs = (node.inputs || []).map((input) => ({ input, match: PROMPT_MEDIA_INPUT_NAME.exec(input.name || "") }))
            .filter(({ match }) => match?.[1] === prefix);
        const highestConnected = Math.max(-1, ...inputs.filter(({ input }) => input.link != null).map(({ match }) => Number(match[2])));
        const target = supported.has(prefix) ? Math.min(maximum - 1, Math.max(0, highestConnected + 1)) : -1;
        for (let index = (node.inputs?.length || 0) - 1; index >= 0; index -= 1) {
            const match = PROMPT_MEDIA_INPUT_NAME.exec(node.inputs[index]?.name || "");
            if (match?.[1] === prefix && Number(match[2]) > target) node.removeInput(index);
        }
        const present = new Set((node.inputs || []).map((input) => input.name));
        for (let index = 0; index <= target; index += 1) {
            const name = `${prefix}_${index}`;
            if (!present.has(name)) node.addInput(name, "IMAGE", { shape: 7 });
        }
    }
    // Inputs are appended by LiteGraph.  Without an explicit reorder, a
    // video socket created at node startup remains between image_0 and later
    // autogrown images.  Preserve the same input objects (and links) while
    // presenting the predictable image_* then video_* layout.
    const mediaInputs = (node.inputs || []).filter((input) => PROMPT_MEDIA_INPUT_NAME.test(input.name || ""));
    const nonMediaInputs = (node.inputs || []).filter((input) => !PROMPT_MEDIA_INPUT_NAME.test(input.name || ""));
    const orderedMedia = ["image", "video"].flatMap((prefix) => mediaInputs
        .filter((input) => input.name.startsWith(`${prefix}_`))
        .sort((left, right) => Number(left.name.slice(prefix.length + 1)) - Number(right.name.slice(prefix.length + 1))));
    const ordered = [...orderedMedia, ...nonMediaInputs];
    if (ordered.some((input, index) => input !== node.inputs[index])) {
        node.inputs.splice(0, node.inputs.length, ...ordered);
        const links = node.graph?.links;
        for (let index = 0; index < node.inputs.length; index += 1) {
            const linkId = node.inputs[index].link;
            const link = links instanceof Map ? links.get(linkId) : links?.[linkId];
            if (link) link.target_slot = index;
        }
    }
    node.setSize?.(node.computeSize?.() || node.size);
    app.graph?.setDirtyCanvas?.(true, true);
}

export function ensureVisibleImageInput(node) {
    const countWidget = node.widgets?.find((widget) => widget.name === "image_count");
    if (!countWidget) throw new Error("image_count 위젯을 찾을 수 없습니다");
    if ((Number(countWidget.value) || 0) < 1) countWidget.value = 1;
    syncImageInputs(node);
}

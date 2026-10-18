import { app } from "../../scripts/app.js";
import { fetchJson, refreshRemoteOptions, refreshRemoteVaeOptions } from "./remote_choices.js";
import { applyPromptProfiles, applyPromptProfilesToVueDefs, refreshPromptProfiles, updateProfileDescription, removeObsoletePromptManagerInput, setupPromptProfileManagerShortcut } from "./remote_profiles.js";
import { normalizeMiniMaxVaeInputs, syncMiniMaxReferenceInputs, syncPromptMediaInputs, ensureVisibleImageInput } from "./remote_sockets.js";
const REFRESH_LABEL = "↻ 원격 CLIP 목록 새로고침";
const VAE_REFRESH_LABEL = "↻ 원격 VAE 목록 새로고침";
const PROFILE_REFRESH_LABEL = "↻ 원격 프로필 목록 새로고침";
let startupPromptProfiles = null;

app.registerExtension({
    name: "RemoteConditioning.ClipLoaderOptions",
    async addCustomNodeDefs(definitions) {
        // Custom-node modules can load after `init` has already run.  This
        // hook is awaited while Comfy has the backend definitions in hand and
        // before either renderer registers them, so it is the reliable place
        // to resolve the initial Vue combo options.
        if (!definitions?.RemotePromptEnhancer) return;
        try {
            startupPromptProfiles = await fetchJson("/remote-prompt-enhancer/profile-options", {
                server_url: definitions.RemotePromptEnhancer.input.required.server_url[1].default,
                timeout_seconds: 5,
            });
            applyPromptProfilesToVueDefs(Object.values(definitions), startupPromptProfiles);
        } catch (error) {
            startupPromptProfiles = null;
            console.warn("[Remote Prompt Enhancer] definition profile lookup failed", error);
        }
    },
    beforeRegisterVueAppNodeDefs(definitions) {
        if (!startupPromptProfiles) return;
        try {
            applyPromptProfilesToVueDefs(definitions, startupPromptProfiles);
        } catch (error) {
            console.warn("[Remote Prompt Enhancer] Vue profile registration failed", error);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "RemoteConditioning" &&
            nodeData?.name !== "RemoteMiniMaxH3ReferenceEncode" &&
            nodeData?.name !== "RemoteMiniMaxH3ImageToVideo" &&
            nodeData?.name !== "RemotePromptEnhancer" &&
            nodeData?.name !== "RemoteVAESelector") return;

        const isPromptEnhancer = nodeData.name === "RemotePromptEnhancer";
        const isMiniMax = nodeData.name === "RemoteMiniMaxH3ReferenceEncode" || nodeData.name === "RemoteMiniMaxH3ImageToVideo";
        const isVaeSelector = nodeData.name === "RemoteVAESelector";
        const isMiniMaxReference = nodeData.name === "RemoteMiniMaxH3ReferenceEncode";
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            if (isPromptEnhancer) removeObsoletePromptManagerInput(node);
            if (isMiniMax) normalizeMiniMaxVaeInputs(node);
            let button = null;
            if (!isVaeSelector) {
                button = node.addWidget("button", isPromptEnhancer ? PROFILE_REFRESH_LABEL : REFRESH_LABEL, "", () => {
                    if (isPromptEnhancer) return refreshPromptProfiles(node, button);
                    return refreshRemoteOptions(node, button);
                }, { serialize: false });
            }
            let vaeButton = null;
            if (isMiniMax || isVaeSelector) {
                vaeButton = node.addWidget("button", VAE_REFRESH_LABEL, "", () => refreshRemoteVaeOptions(node, vaeButton), { serialize: false });
                vaeButton.name = isVaeSelector
                    ? `${VAE_REFRESH_LABEL} — MiniMax와 같은 server_url 사용`
                    : VAE_REFRESH_LABEL;
            }
            if (isPromptEnhancer) {
                setupPromptProfileManagerShortcut(node);
                const descriptionButton = node.addWidget("button", "ℹ 프로필 설명을 보려면 목록을 새로고침", "", () => {}, { serialize: false });
                node._remotePromptDescriptionButton = descriptionButton;
                // `addCustomNodeDefs` runs before either renderer creates its
                // node instance.  Carry that same verified profile payload
                // into the instance so the first media socket is visible
                // immediately; a manual refresh still replaces it for a
                // non-default peer endpoint.
                if (startupPromptProfiles) {
                    try {
                        applyPromptProfiles(node, startupPromptProfiles);
                    } catch (error) {
                        console.warn("[Remote Prompt Enhancer] initial profile application failed", error);
                    }
                }
                const profileWidget = node.widgets?.find((widget) => widget.name === "profile_id");
                if (profileWidget) {
                    const onProfileChanged = profileWidget.callback;
                    profileWidget.callback = function () {
                        const callbackResult = onProfileChanged?.apply(this, arguments);
                        updateProfileDescription(node);
                        queueMicrotask(() => syncPromptMediaInputs(node));
                        return callbackResult;
                    };
                }
                updateProfileDescription(node);
                const onResize = node.onResize;
                node.onResize = function () {
                    const result = onResize?.apply(this, arguments);
                    updateProfileDescription(node);
                    return result;
                };
                const onConnectionsChange = node.onConnectionsChange;
                node.onConnectionsChange = function () {
                    const callbackResult = onConnectionsChange?.apply(this, arguments);
                    queueMicrotask(() => syncPromptMediaInputs(node));
                    return callbackResult;
                };
                queueMicrotask(() => syncPromptMediaInputs(node));
            }
            if (isMiniMaxReference) {
                const onConnectionsChange = node.onConnectionsChange;
                node.onConnectionsChange = function () {
                    const callbackResult = onConnectionsChange?.apply(this, arguments);
                    queueMicrotask(() => syncMiniMaxReferenceInputs(node));
                    return callbackResult;
                };
            } else if (nodeData.name === "RemoteConditioning") {
                const countWidget = node.widgets?.find((widget) => widget.name === "image_count");
                if (countWidget) {
                    const onCountChanged = countWidget.callback;
                    countWidget.callback = function () { const callbackResult = onCountChanged?.apply(this, arguments); ensureVisibleImageInput(node); return callbackResult; };
                }
            }
            if (isMiniMaxReference) syncMiniMaxReferenceInputs(node);
            else if (nodeData.name === "RemoteConditioning") ensureVisibleImageInput(node);
            node.setSize?.(node.computeSize?.() || node.size);
            return result;
        };
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            if (isMiniMax) normalizeMiniMaxVaeInputs(this);
            if (nodeData.name === "RemoteMiniMaxH3ReferenceEncode") queueMicrotask(() => {
                normalizeMiniMaxVaeInputs(this);
                syncMiniMaxReferenceInputs(this);
            });
            else if (nodeData.name === "RemoteMiniMaxH3ImageToVideo") queueMicrotask(() => normalizeMiniMaxVaeInputs(this));
            else if (nodeData.name === "RemotePromptEnhancer") queueMicrotask(() => {
                removeObsoletePromptManagerInput(this);
                syncPromptMediaInputs(this);
                updateProfileDescription(this);
            });
            else if (nodeData.name === "RemoteConditioning") queueMicrotask(() => ensureVisibleImageInput(this));
            return result;
        };
    },
});

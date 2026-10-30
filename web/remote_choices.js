import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const OPTION_WIDGET_NAMES = ["clip_name", "clip_type", "device"];
const OPTION_WIDGET_ALIASES = {
    clip_name: ["clip_name", "remote_text_encoder"],
    clip_type: ["clip_type"],
    device: ["device"],
};


const VAE_WIDGET_NAMES = ["vae_name", "audio_vae_name"];


export function findOptionWidget(node, name) {
    const candidates = OPTION_WIDGET_ALIASES[name] || [name];
    return node.widgets?.find((widget) => candidates.includes(widget.name));
}

export function cleanOptions(values, name) {
    if (!Array.isArray(values)) {
        throw new Error(`${name} 선택 목록이 없습니다`);
    }
    if (values.some((value) => typeof value !== "string" || !value.trim())) {
        throw new Error(`${name} 선택 목록에 올바르지 않은 값이 있습니다`);
    }
    const cleaned = [...new Set(values)];
    if (!cleaned.length) {
        throw new Error(`${name} 선택 목록이 비어 있습니다`);
    }
    return cleaned;
}

export function resolvePeerModelChoice(current, choices) {
    if (choices.includes(current)) return current;
    if (typeof current !== "string" || !current) return current ?? "";
    const matches = choices.filter(name => name.replaceAll("\\", "/") === current.replaceAll("\\", "/"));
    return matches.length === 1 ? matches[0] : current;
}

export function normalizeRemoteOptions(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error("원격 CLIP 응답 형식이 올바르지 않습니다");
    }
    return Object.fromEntries(OPTION_WIDGET_NAMES.map((name) => [name, cleanOptions(payload[name], name)]));
}

export function normalizeRemoteVaeOptions(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error("원격 VAE 응답 형식이 올바르지 않습니다");
    }
    return { vae_name: cleanOptions(payload.vae_name, "vae_name") };
}

export function reconcileRemoteOptions(current, remote) {
    const normalized = normalizeRemoteOptions(remote);
    return Object.fromEntries(
        OPTION_WIDGET_NAMES.map((name) => [
            name,
            resolvePeerModelChoice(current?.[name], normalized[name]),
        ]),
    );
}

export function applyRemoteVaeOptions(node, remote) {
    const normalized = normalizeRemoteVaeOptions(remote);
    const widgets = VAE_WIDGET_NAMES
        .map((name) => node.widgets?.find((widget) => widget.name === name))
        .filter(Boolean);
    if (!widgets.length) {
        throw new Error("vae_name 위젯을 찾을 수 없습니다");
    }
    for (const widget of widgets) {
        applyChoiceWidget(widget, normalized.vae_name);
        widget.callback?.(widget.value, app.canvas, node, app.canvas?.graph_mouse);
    }
    app.graph?.setDirtyCanvas?.(true, true);
    return normalized.vae_name.length;
}

export function applyRemoteOptions(node, remote) {
    const widgets = Object.fromEntries(
        OPTION_WIDGET_NAMES.map((name) => [name, findOptionWidget(node, name)]),
    );
    const missing = OPTION_WIDGET_NAMES.find((name) => !widgets[name]);
    if (missing) {
        throw new Error(`${missing} 위젯을 찾을 수 없습니다`);
    }

    // Validate and reconcile every field before mutating any widget.
    const normalized = normalizeRemoteOptions(remote);
    const selected = reconcileRemoteOptions(
        Object.fromEntries(OPTION_WIDGET_NAMES.map((name) => [name, widgets[name].value])),
        normalized,
    );

    for (const name of OPTION_WIDGET_NAMES) {
        const widget = widgets[name];
        widget.value = selected[name];
        applyChoiceWidget(widget, normalized[name]);
    }
    for (const name of OPTION_WIDGET_NAMES) {
        const widget = widgets[name];
        widget.callback?.(widget.value, app.canvas, node, app.canvas?.graph_mouse);
    }
    app.graph?.setDirtyCanvas?.(true, true);
    return normalized.clip_name.length;
}

export async function readError(response) {
    try {
        const payload = await response.json();
        const error = payload?.error;
        if (typeof error === "string") return error;
        return error?.message || payload?.message || `HTTP ${response.status}`;
    } catch (_error) {
        return `HTTP ${response.status}`;
    }
}

export function applyChoiceWidget(widget, choices) {
    widget.value = resolvePeerModelChoice(widget.value, choices);
    widget._remotePeerChoices = choices;
    widget.options = { ...(widget.options || {}), values: choices.includes(widget.value) ? choices : [widget.value, ...choices] };
    widget.type = "combo";
    if (!widget._remoteChoiceCallback) {
        widget._remoteChoiceHelp = widget.options.tooltip || "";
        const previous = widget.callback;
        widget.callback = function () {
            const result = previous?.apply(this, arguments);
            updateChoiceHelp(widget);
            for (const refresh of widget._remoteChoiceStatus?.values() || []) refresh();
            return result;
        };
        widget._remoteChoiceCallback = true;
    }
    updateChoiceHelp(widget);
}

function updateChoiceHelp(widget) {
    const missing = widget.value && !widget._remotePeerChoices?.includes(widget.value);
    widget.options.tooltip = [widget._remoteChoiceHelp, missing ? "현재 선택 미확인 · 목록에서 다시 선택" : ""].filter(Boolean).join("\n");
}

export function choiceStatus(node, names) {
    const widgets = names.map(name => findOptionWidget(node, name)).filter(Boolean);
    if (widgets.some(widget => !widget.value)) return "empty";
    return widgets.some(widget => !widget._remotePeerChoices?.includes(widget.value)) ? "missing" : "ready";
}

function trackServer(node, serverWidget) {
    if (!node._remoteChoiceServer) {
        node._remoteChoiceServer = { value: serverWidget.value, revision: 0, buttons: new Map() };
        const previous = serverWidget.callback;
        serverWidget.callback = function () {
            const result = previous?.apply(this, arguments);
            const state = node._remoteChoiceServer;
            if (state.value !== serverWidget.value) {
                state.value = serverWidget.value;
                state.revision++;
                for (const [button, config] of state.buttons) {
                    config.label(config.serverChangedLabel);
                    button.options = { ...(button.options || {}), tooltip: "서버가 바뀌었습니다. 목록을 새로고침하세요. 기존 선택 유지." };
                }
                app.graph?.setDirtyCanvas?.(true, true);
            }
            return result;
        };
    }
    const state = node._remoteChoiceServer;
    if (state.value !== serverWidget.value) {
        state.value = serverWidget.value;
        state.revision++;
    }
    return state;
}

export function remoteServerRevision(node) {
    const server = node.widgets?.find(widget => widget.name === "server_url");
    return server ? trackServer(node, server).revision : 0;
}

export async function refreshRemoteChoiceOptions(node, button, config) {
    const setLabel = config.setLabel || ((label) => { button.name = label; });
    const serverWidget = node.widgets?.find((widget) => widget.name === "server_url");
    const timeoutWidget = node.widgets?.find((widget) => widget.name === "timeout_seconds");
    if (!serverWidget || !timeoutWidget) {
        setLabel(config.missingSettingsLabel);
        return;
    }
    const state = trackServer(node, serverWidget);
    state.buttons.set(button, { ...config, label: setLabel });
    const previous = button._remoteChoiceRequest;
    if (previous?.server === state.value && previous.revision === state.revision) return;
    const request = { server: state.value, revision: state.revision };
    button._remoteChoiceRequest = request;
    const current = () => button._remoteChoiceRequest === request && serverWidget.value === request.server && state.revision === request.revision;
    button[config.loadingKey] = true;
    setLabel(config.loadingLabel);
    button.options = { ...(button.options || {}), tooltip: "목록 조회 중 · 기존 선택 유지" };
    app.graph?.setDirtyCanvas?.(true, true);
    try {
        const response = await api.fetchApi(config.path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ server_url: request.server, timeout_seconds: Math.min(30, Math.max(5, Number(timeoutWidget.value) || 5)) }),
        });
        if (!response.ok) throw new Error(await readError(response));
        const payload = await response.json();
        if (!current()) return;
        const count = config.apply(node, payload);
        button.options.tooltip = `목록 ${count}개 · 기존 선택 유지`;
        setLabel(config.successLabel(count));
        for (const name of config.widgetNames || []) {
            const widget = findOptionWidget(node, name);
            if (!widget) continue;
            widget._remoteChoiceStatus ??= new Map();
            widget._remoteChoiceStatus.set(button, () => {
                if (!button._remoteChoiceRequest && serverWidget.value === request.server && state.revision === request.revision) {
                    setLabel(config.successLabel(count));
                    app.graph?.setDirtyCanvas?.(true, true);
                }
            });
        }
    } catch (error) {
        if (!current()) return;
        const detail = error instanceof Error ? error.message : String(error);
        button.options.tooltip = `조회 실패 · 기존 선택 유지\n${detail}`;
        const label = config.failureLabel(detail);
        setLabel(label);
        if (config.notify !== false) app.extensionManager?.toast?.add({ severity: "error", summary: label,
            detail: `${detail.slice(0, 180)} 서버 주소 확인 후 재시도하세요. 기존 선택 유지.`, life: 7000 });
        console.warn(config.warningPrefix, error);
    } finally {
        if (button._remoteChoiceRequest === request) {
            if (!current()) setLabel(config.serverChangedLabel);
            delete button._remoteChoiceRequest;
            button[config.loadingKey] = false;
        }
        app.graph?.setDirtyCanvas?.(true, true);
    }
}

export async function refreshRemoteOptions(node, button) {
    return refreshRemoteChoiceOptions(node, button, {
        loadingKey: "_remoteConditioningLoading",
        path: "/remote-conditioning/clip-loader-options",
        widgetNames: OPTION_WIDGET_NAMES,
        missingSettingsLabel: "서버 설정 확인",
        loadingLabel: "CLIP 조회 중…",
        serverChangedLabel: "서버 변경됨 · 새로고침",
        successLabel: (count) => choiceStatus(node, OPTION_WIDGET_NAMES) === "ready" ? `CLIP ${count}개 · 새로고침` : "다시 선택 · CLIP 새로고침",
        failureLabel: () => "조회 실패 · CLIP 재시도",
        warningPrefix: "[Remote Conditioning] CLIP 목록 조회 실패",
        apply: applyRemoteOptions,
    });
}

export async function refreshRemoteVaeOptions(node, button) {
    return refreshRemoteChoiceOptions(node, button, {
        loadingKey: "_remoteVaeLoading",
        path: "/remote-conditioning/vae-loader-options",
        widgetNames: VAE_WIDGET_NAMES,
        missingSettingsLabel: "서버 설정 확인",
        loadingLabel: "VAE 조회 중…",
        serverChangedLabel: "서버 변경됨 · 새로고침",
        successLabel: (count) => choiceStatus(node, VAE_WIDGET_NAMES) === "ready" ? `VAE ${count}개 · 새로고침` : "다시 선택 · VAE 새로고침",
        failureLabel: () => "조회 실패 · VAE 재시도",
        warningPrefix: "[Remote Conditioning] VAE 목록 조회 실패",
        apply: applyRemoteVaeOptions,
    });
}

export function remoteRequestConfig(node) {
    const serverWidget = node.widgets?.find((widget) => widget.name === "server_url");
    const timeoutWidget = node.widgets?.find((widget) => widget.name === "timeout_seconds");
    if (!serverWidget || !timeoutWidget) throw new Error("원격 설정 위젯이 없습니다");
    return {
        server_url: serverWidget.value,
        timeout_seconds: Math.min(30, Math.max(5, Number(timeoutWidget.value) || 5)),
    };
}

export async function fetchJson(path, payload) {
    const response = await api.fetchApi(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await readError(response));
    return response.json();
}

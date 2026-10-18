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
    const cleaned = [...new Set(values.map((value) => value.trim()))];
    if (!cleaned.length) {
        throw new Error(`${name} 선택 목록이 비어 있습니다`);
    }
    return cleaned;
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
            normalized[name].includes(current?.[name]) ? current[name] : normalized[name][0],
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
        widget.options = { ...(widget.options || {}), values: normalized.vae_name };
        if (!normalized.vae_name.includes(widget.value)) widget.value = normalized.vae_name[0];
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
        widget.options = { ...(widget.options || {}) };
        widget.options.values = normalized[name];
        widget.type = "combo";
        widget.value = selected[name];
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

export async function refreshRemoteChoiceOptions(node, button, config) {
    if (button[config.loadingKey]) return;
    const serverWidget = node.widgets?.find((widget) => widget.name === "server_url");
    const timeoutWidget = node.widgets?.find((widget) => widget.name === "timeout_seconds");
    if (!serverWidget || !timeoutWidget) {
        button.name = config.missingSettingsLabel;
        return;
    }

    button[config.loadingKey] = true;
    button.name = config.loadingLabel;
    app.graph?.setDirtyCanvas?.(true, true);
    try {
        const lookupTimeout = Math.min(30, Math.max(5, Number(timeoutWidget.value) || 5));
        const requestedServerUrl = serverWidget.value;
        const response = await api.fetchApi(config.path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                server_url: requestedServerUrl,
                timeout_seconds: lookupTimeout,
            }),
        });
        if (!response.ok) {
            throw new Error(await readError(response));
        }
        const payload = await response.json();
        if (serverWidget.value !== requestedServerUrl) {
            button.name = config.serverChangedLabel;
            return;
        }
        const count = config.apply(node, payload);
        button.name = config.successLabel(count);
    } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        button.name = config.failureLabel(detail);
        console.warn(config.warningPrefix, error);
    } finally {
        button[config.loadingKey] = false;
        app.graph?.setDirtyCanvas?.(true, true);
    }
}

export async function refreshRemoteOptions(node, button) {
    return refreshRemoteChoiceOptions(node, button, {
        loadingKey: "_remoteConditioningLoading",
        path: "/remote-conditioning/clip-loader-options",
        missingSettingsLabel: "⚠ 원격 설정 위젯 없음 — 기존 선택 유지",
        loadingLabel: "… 원격 CLIP 목록 조회 중",
        serverChangedLabel: "↻ 원격 CLIP 목록 새로고침 — server_url 변경됨",
        successLabel: (count) => `✓ 원격 CLIP ${count}개 — 다시 조회`,
        failureLabel: (detail) => `⚠ 조회 실패: ${detail} — 기존 선택 유지`,
        warningPrefix: "[Remote Conditioning] CLIP 목록 조회 실패",
        apply: applyRemoteOptions,
    });
}

export async function refreshRemoteVaeOptions(node, button) {
    return refreshRemoteChoiceOptions(node, button, {
        loadingKey: "_remoteVaeLoading",
        path: "/remote-conditioning/vae-loader-options",
        missingSettingsLabel: "⚠ 원격 설정 위젯 없음 — 기존 VAE 선택 유지",
        loadingLabel: "… 원격 VAE 목록 조회 중",
        serverChangedLabel: "↻ 원격 VAE 목록 새로고침 — server_url 변경됨",
        successLabel: (count) => `✓ 원격 VAE ${count}개 — 다시 조회`,
        failureLabel: (detail) => `⚠ VAE 조회 실패: ${detail} — 기존 선택 유지`,
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

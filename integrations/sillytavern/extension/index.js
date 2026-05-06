const MODULE_NAME = 'atagia_memory';
const DISPLAY_NAME = 'Atagia Memory';
const TRANSPORT_ID_PREFIX = '__atagia_b64_';
const SAFE_ID_PATTERN = /^[A-Za-z0-9_:-][A-Za-z0-9_.:-]*$/;

const defaultSettings = Object.freeze({
    enabled: false,
    baseUrl: 'http://127.0.0.1:8100',
    apiKey: '',
    userId: 'sillytavern-user',
    platformId: 'sillytavern',
    conversationPrefix: 'sillytavern',
    mode: 'companion',
    debug: false,
    lastPreview: '',
    lastStatus: 'Not connected',
});

function getContext() {
    return globalThis.SillyTavern?.getContext?.();
}

function settings() {
    const context = getContext();
    if (!context.extensionSettings[MODULE_NAME]) {
        context.extensionSettings[MODULE_NAME] = structuredClone(defaultSettings);
    }
    const current = context.extensionSettings[MODULE_NAME];
    for (const key of Object.keys(defaultSettings)) {
        if (!Object.hasOwn(current, key)) {
            current[key] = defaultSettings[key];
        }
    }
    return current;
}

function saveSettings() {
    getContext().saveSettingsDebounced();
}

function cleanBaseUrl(value) {
    return String(value || '').replace(/\/+$/, '');
}

function escapeAttribute(value) {
    return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('"', '&quot;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
}

function escapeText(value) {
    return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
}

function base64UrlEncode(value) {
    const bytes = new TextEncoder().encode(String(value));
    let binary = '';
    for (const byte of bytes) {
        binary += String.fromCharCode(byte);
    }
    return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}

function transportId(value) {
    const text = String(value || '');
    if (
        text !== '.'
        && text !== '..'
        && !text.startsWith(TRANSPORT_ID_PREFIX)
        && SAFE_ID_PATTERN.test(text)
    ) {
        return text;
    }
    return `${TRANSPORT_ID_PREFIX}${base64UrlEncode(text)}`;
}

function firstNonEmpty(...values) {
    for (const value of values) {
        if (typeof value === 'string' && value.trim()) {
            return value.trim();
        }
    }
    return null;
}

function currentConversationId() {
    const context = getContext();
    const current = settings();
    const persisted = firstNonEmpty(context.chatMetadata?.atagia_conversation_id);
    if (persisted) {
        return persisted;
    }
    const raw = firstNonEmpty(
        context.chatMetadata?.chat_id,
        context.chatId,
        context.groupId,
    );
    if (!raw) {
        return null;
    }
    const generated = transportId(JSON.stringify({
        prefix: String(current.conversationPrefix || 'sillytavern'),
        chatId: String(raw),
    }));
    if (context.chatMetadata && typeof context.chatMetadata === 'object') {
        context.chatMetadata.atagia_conversation_id = generated;
        context.saveMetadata?.();
    }
    return generated;
}

function latestUserMessage(chat) {
    for (let index = chat.length - 1; index >= 0; index -= 1) {
        const entry = chat[index];
        if (entry?.is_user && entry?.mes) {
            return String(entry.mes);
        }
    }
    return '';
}

async function atagiaFetch(path, options = {}) {
    const current = settings();
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${current.apiKey}`,
        'X-Atagia-User-Id': current.userId,
        'X-Atagia-Platform-Id': current.platformId,
        ...options.headers,
    };
    const conversationId = currentConversationId();
    if (conversationId) {
        headers['X-Atagia-Conversation-Id'] = conversationId;
    }
    const response = await fetch(`${cleanBaseUrl(current.baseUrl)}${path}`, {
        ...options,
        headers,
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Atagia ${response.status}: ${text}`);
    }
    if (response.status === 204) {
        return null;
    }
    return response.json();
}

function injectionMessage(systemPrompt) {
    return {
        is_user: false,
        name: 'Atagia Memory',
        send_date: Date.now(),
        mes: [
            '[ATAGIA MEMORY CONTEXT - INTERNAL]',
            'Use this memory context for continuity. Do not quote this block verbatim.',
            '',
            systemPrompt,
            '[/ATAGIA MEMORY CONTEXT]',
        ].join('\n'),
    };
}

async function fetchContextForTurn(chat) {
    const current = settings();
    const messageText = latestUserMessage(chat);
    if (!messageText) {
        return null;
    }
    const conversationId = currentConversationId();
    if (!conversationId) {
        return null;
    }
    return atagiaFetch(`/v1/conversations/${encodeURIComponent(conversationId)}/context`, {
        method: 'POST',
        body: JSON.stringify({
            user_id: current.userId,
            message_text: messageText,
            platform_id: current.platformId,
            mode: current.mode || null,
        }),
    });
}

async function recordAssistantResponse(messageText) {
    const current = settings();
    if (!current.enabled || !messageText) {
        return;
    }
    const conversationId = currentConversationId();
    if (!conversationId) {
        return;
    }
    await atagiaFetch(`/v1/conversations/${encodeURIComponent(conversationId)}/responses`, {
        method: 'POST',
        body: JSON.stringify({
            user_id: current.userId,
            text: messageText,
            platform_id: current.platformId,
            mode: current.mode || null,
        }),
    });
}

globalThis.atagiaMemoryInterceptor = async function atagiaMemoryInterceptor(chat, contextSize, abort, type) {
    const current = settings();
    if (!current.enabled || type === 'quiet') {
        return;
    }
    try {
        const context = await fetchContextForTurn(chat);
        const systemPrompt = String(context?.system_prompt || '').trim();
        if (!systemPrompt) {
            current.lastStatus = 'No Atagia context returned';
            current.lastPreview = '';
            saveSettings();
            return;
        }
        current.lastStatus = 'Context injected';
        current.lastPreview = systemPrompt;
        saveSettings();
        chat.splice(Math.max(chat.length - 1, 0), 0, injectionMessage(systemPrompt));
    } catch (error) {
        current.lastStatus = `Atagia unavailable: ${error.message}`;
        saveSettings();
        if (current.debug) {
            console.warn('[Atagia Memory] failed open', error);
        }
    }
};

async function handleMessageReceived(data) {
    const text = String(data?.mes || data?.message || '').trim();
    if (!text) {
        return;
    }
    try {
        await recordAssistantResponse(text);
    } catch (error) {
        const current = settings();
        current.lastStatus = `Response persistence failed: ${error.message}`;
        saveSettings();
        if (current.debug) {
            console.warn('[Atagia Memory] response persistence failed', error);
        }
    }
}

function renderSettings() {
    const context = getContext();
    const current = settings();
    const html = `
        <div class="atagia-memory-settings">
            <div class="inline-drawer">
                <div class="inline-drawer-toggle inline-drawer-header">
                    <b>${DISPLAY_NAME}</b>
                    <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
                </div>
                <div class="inline-drawer-content">
                    <label class="checkbox_label">
                        <input id="atagia_memory_enabled" type="checkbox" ${current.enabled ? 'checked' : ''}>
                        <span>Enable Atagia memory</span>
                    </label>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_base_url">Atagia base URL</label>
                        <input id="atagia_memory_base_url" type="text" value="${escapeAttribute(current.baseUrl)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_api_key">Service API key</label>
                        <input id="atagia_memory_api_key" type="password" value="${escapeAttribute(current.apiKey)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_user_id">User ID</label>
                        <input id="atagia_memory_user_id" type="text" value="${escapeAttribute(current.userId)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_platform_id">Platform ID</label>
                        <input id="atagia_memory_platform_id" type="text" value="${escapeAttribute(current.platformId)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_conversation_prefix">Conversation prefix</label>
                        <input id="atagia_memory_conversation_prefix" type="text" value="${escapeAttribute(current.conversationPrefix)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_mode">Mode</label>
                        <input id="atagia_memory_mode" type="text" value="${escapeAttribute(current.mode)}">
                    </div>
                    <label class="checkbox_label">
                        <input id="atagia_memory_debug" type="checkbox" ${current.debug ? 'checked' : ''}>
                        <span>Debug logging</span>
                    </label>
                    <div class="atagia-memory-actions">
                        <button id="atagia_memory_test" type="button">Test connection</button>
                    </div>
                    <small id="atagia_memory_status">${escapeText(current.lastStatus)}</small>
                    <pre class="atagia-memory-preview" id="atagia_memory_preview">${escapeText(current.lastPreview)}</pre>
                </div>
            </div>
        </div>`;
    $('#extensions_settings2').append(html);

    $('#atagia_memory_enabled').on('change', function onChange() {
        current.enabled = this.checked;
        saveSettings();
    });
    $('#atagia_memory_base_url').on('input', function onInput() {
        current.baseUrl = this.value;
        saveSettings();
    });
    $('#atagia_memory_api_key').on('input', function onInput() {
        current.apiKey = this.value;
        saveSettings();
    });
    $('#atagia_memory_user_id').on('input', function onInput() {
        current.userId = this.value;
        saveSettings();
    });
    $('#atagia_memory_platform_id').on('input', function onInput() {
        current.platformId = this.value;
        saveSettings();
    });
    $('#atagia_memory_conversation_prefix').on('input', function onInput() {
        current.conversationPrefix = this.value;
        saveSettings();
    });
    $('#atagia_memory_mode').on('input', function onInput() {
        current.mode = this.value;
        saveSettings();
    });
    $('#atagia_memory_debug').on('change', function onChange() {
        current.debug = this.checked;
        saveSettings();
    });
    $('#atagia_memory_test').on('click', async () => {
        try {
            await atagiaFetch('/v1/models', { method: 'GET' });
            current.lastStatus = 'Atagia connection successful';
        } catch (error) {
            current.lastStatus = `Atagia connection failed: ${error.message}`;
        }
        $('#atagia_memory_status').text(current.lastStatus);
        saveSettings();
    });

    context.eventSource.on(context.event_types.MESSAGE_RECEIVED, handleMessageReceived);
}

export async function onActivate() {
    const context = getContext();
    context.eventSource.on(context.event_types.APP_READY, renderSettings);
}

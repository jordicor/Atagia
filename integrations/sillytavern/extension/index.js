const MODULE_NAME = 'atagia_memory';
const DISPLAY_NAME = 'Atagia Memory';
const TRANSPORT_ID_PREFIX = '__atagia_b64_';
const SAFE_ID_PATTERN = /^[A-Za-z0-9_:-][A-Za-z0-9_.:-]*$/;

const defaultSettings = Object.freeze({
    enabled: false,
    baseUrl: 'http://127.0.0.1:8100',
    apiKey: '',
    userId: 'sillytavern-user',
    userPersonaId: '',
    platformId: 'sillytavern',
    characterId: '',
    conversationPrefix: 'sillytavern',
    mode: 'companion',
    memoryPrivacyMode: 'balanced',
    debug: false,
    lastPreview: '',
    lastStatus: 'Not connected',
    lastRequest: '',
    lastRequestMessageId: '',
    lastError: '',
});

function getContext() {
    return globalThis.SillyTavern?.getContext?.();
}

function settings() {
    const context = getContext();
    if (!context) {
        throw new Error('SillyTavern context is unavailable');
    }
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
    getContext()?.saveSettingsDebounced?.();
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

function ensureChatMetadata(context) {
    if (!context.chatMetadata || typeof context.chatMetadata !== 'object') {
        context.chatMetadata = {};
    }
    return context.chatMetadata;
}

function currentConversationId() {
    const context = getContext();
    const current = settings();
    const metadata = ensureChatMetadata(context);
    const persisted = firstNonEmpty(metadata.atagia_conversation_id);
    if (persisted) {
        return persisted;
    }
    const raw = firstNonEmpty(metadata.chat_id, context.chatId, context.groupId);
    if (!raw) {
        return null;
    }
    const generated = transportId(JSON.stringify({
        prefix: String(current.conversationPrefix || 'sillytavern'),
        chatId: String(raw),
    }));
    metadata.atagia_conversation_id = generated;
    context.saveMetadata?.();
    return generated;
}

function currentUserPersonaId() {
    const context = getContext();
    const current = settings();
    const raw = firstNonEmpty(current.userPersonaId, context?.name1);
    return raw ? transportId(raw) : null;
}

function currentCharacterId() {
    const context = getContext();
    const current = settings();
    const raw = firstNonEmpty(
        current.characterId,
        context?.characterId,
        context?.character?.name,
        context?.name2,
    );
    return raw ? transportId(raw) : null;
}

function messageMetadata(entry) {
    if (!entry || typeof entry !== 'object') {
        return {};
    }
    if (entry.extra && typeof entry.extra === 'object') {
        return entry.extra;
    }
    if (entry.metadata && typeof entry.metadata === 'object') {
        return entry.metadata;
    }
    return {};
}

function contentHash(text) {
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
        hash ^= text.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function stableSourceSeq(index, role, text, variant = '') {
    const roleOffset = role === 'assistant' ? 50000 : 1;
    const contentOffset = contentHash(`${role}:${variant}:${text}`) % 49999;
    return ((index + 1) * 100000) + roleOffset + contentOffset;
}

function stableMessageId(conversationId, role, entry, index, text, variant = '') {
    const metadata = messageMetadata(entry);
    const explicit = firstNonEmpty(
        metadata.atagia_message_id,
        metadata.message_id,
        entry?.swipe_id,
    );
    if (explicit) {
        return transportId(explicit);
    }
    return transportId(JSON.stringify({
        conversationId,
        role,
        index,
        sendDate: String(entry?.send_date || entry?.sendDate || ''),
        name: String(entry?.name || ''),
        variant,
        textHash: contentHash(text),
    }));
}

function latestUserMessageInfo(chat) {
    const conversationId = currentConversationId();
    if (!conversationId || !Array.isArray(chat)) {
        return null;
    }
    for (let index = chat.length - 1; index >= 0; index -= 1) {
        const entry = chat[index];
        const text = String(entry?.mes || '').trim();
        if (entry?.is_user && text) {
            return {
                text,
                sourceSeq: stableSourceSeq(index, 'user', text),
                messageId: stableMessageId(conversationId, 'user', entry, index, text),
            };
        }
    }
    return null;
}

function assistantMessageInfo(data) {
    const context = getContext();
    const conversationId = currentConversationId();
    const text = String(data?.mes || data?.message || '').trim();
    if (!conversationId || !text) {
        return null;
    }
    const chat = Array.isArray(context?.chat) ? context.chat : [];
    for (let index = chat.length - 1; index >= 0; index -= 1) {
        const entry = chat[index];
        if (entry?.is_user) {
            continue;
        }
        const entryText = String(entry?.mes || '').trim();
        if (entryText === text) {
            const variant = String(entry?.swipe_id ?? entry?.swipeId ?? entry?.swipe_index ?? '');
            return {
                text,
                sourceSeq: stableSourceSeq(index, 'assistant', text, variant),
                messageId: stableMessageId(conversationId, 'assistant', entry, index, text, variant),
            };
        }
    }
    const index = chat.length;
    return {
        text,
        sourceSeq: stableSourceSeq(index, 'assistant', text),
        messageId: stableMessageId(conversationId, 'assistant', data, index, text),
    };
}

function identityPayload() {
    const current = settings();
    return {
        user_id: current.userId,
        platform_id: current.platformId,
        character_id: currentCharacterId(),
        user_persona_id: currentUserPersonaId(),
        mode: current.mode || null,
        memory_privacy_mode: current.memoryPrivacyMode || null,
    };
}

function rememberLastRequest(path, payload, messageId) {
    const current = settings();
    current.lastRequestMessageId = messageId || '';
    current.lastRequest = JSON.stringify({
        path,
        message_id: messageId || null,
        conversation_id: currentConversationId(),
        platform_id: current.platformId,
        payload,
    }, null, 2);
    saveSettings();
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

function buildMemoryBlock(systemPrompt) {
    return [
        '[ATAGIA MEMORY CONTEXT - INTERNAL]',
        'Use this memory context for continuity. Do not quote this block verbatim.',
        '',
        systemPrompt,
        '[/ATAGIA MEMORY CONTEXT]',
    ].join('\n');
}

function applyExtensionPrompt(systemPrompt) {
    const context = getContext();
    const block = buildMemoryBlock(systemPrompt);
    if (typeof context?.setExtensionPrompt === 'function') {
        context.setExtensionPrompt(MODULE_NAME, block, 0, 1, false);
        return true;
    }
    if (typeof globalThis.setExtensionPrompt === 'function') {
        globalThis.setExtensionPrompt(MODULE_NAME, block, 0, 1, false);
        return true;
    }
    return false;
}

function clearExtensionPrompt() {
    const context = getContext();
    if (typeof context?.setExtensionPrompt === 'function') {
        context.setExtensionPrompt(MODULE_NAME, '', 0, 1, false);
    } else if (typeof globalThis.setExtensionPrompt === 'function') {
        globalThis.setExtensionPrompt(MODULE_NAME, '', 0, 1, false);
    }
}

async function fetchContextForTurn(chat) {
    const info = latestUserMessageInfo(chat);
    if (!info) {
        return null;
    }
    const conversationId = currentConversationId();
    const payload = {
        ...identityPayload(),
        message_text: info.text,
        message_id: info.messageId,
        source_seq: info.sourceSeq,
        ingest_origin: 'live_turn',
        confirmation_strategy: 'live_prompt_allowed',
    };
    rememberLastRequest(`/v1/conversations/${conversationId}/context`, payload, info.messageId);
    return atagiaFetch(`/v1/conversations/${encodeURIComponent(conversationId)}/context`, {
        method: 'POST',
        headers: {
            'X-Atagia-Message-Id': info.messageId,
            'X-Atagia-Source-Seq': String(info.sourceSeq),
            'X-Atagia-Ingest-Origin': 'live_turn',
            'X-Atagia-Confirmation-Strategy': 'live_prompt_allowed',
            'X-Atagia-Memory-Privacy-Mode': settings().memoryPrivacyMode,
        },
        body: JSON.stringify(payload),
    });
}

async function recordAssistantResponse(data) {
    const current = settings();
    const info = assistantMessageInfo(data);
    if (!current.enabled || !info) {
        return;
    }
    const conversationId = currentConversationId();
    const payload = {
        ...identityPayload(),
        text: info.text,
        message_id: info.messageId,
        source_seq: info.sourceSeq,
        ingest_origin: 'live_turn',
        confirmation_strategy: 'live_prompt_allowed',
    };
    rememberLastRequest(`/v1/conversations/${conversationId}/responses`, payload, info.messageId);
    await atagiaFetch(`/v1/conversations/${encodeURIComponent(conversationId)}/responses`, {
        method: 'POST',
        headers: {
            'X-Atagia-Response-Message-Id': info.messageId,
            'X-Atagia-Response-Source-Seq': String(info.sourceSeq),
            'X-Atagia-Ingest-Origin': 'live_turn',
            'X-Atagia-Confirmation-Strategy': 'live_prompt_allowed',
            'X-Atagia-Memory-Privacy-Mode': current.memoryPrivacyMode,
        },
        body: JSON.stringify(payload),
    });
}

globalThis.atagiaMemoryInterceptor = async function atagiaMemoryInterceptor(chat, contextSize, abort, type) {
    const current = settings();
    if (!current.enabled || type === 'quiet') {
        clearExtensionPrompt();
        return;
    }
    try {
        const context = await fetchContextForTurn(chat);
        const systemPrompt = String(context?.system_prompt || '').trim();
        if (!systemPrompt) {
            clearExtensionPrompt();
            current.lastStatus = 'No Atagia context returned';
            current.lastPreview = '';
            current.lastError = '';
            saveSettings();
            return;
        }
        const injected = applyExtensionPrompt(systemPrompt);
        current.lastStatus = injected
            ? 'Context injected'
            : 'Prompt injection API unavailable; failed open';
        current.lastPreview = systemPrompt;
        current.lastError = injected ? '' : 'setExtensionPrompt is unavailable';
        saveSettings();
    } catch (error) {
        clearExtensionPrompt();
        current.lastStatus = `Atagia unavailable: ${error.message}`;
        current.lastError = String(error.message || error);
        saveSettings();
        if (current.debug) {
            console.warn('[Atagia Memory] failed open', error);
        }
    }
};

async function handleMessageReceived(data) {
    try {
        await recordAssistantResponse(data);
        const current = settings();
        if (current.enabled) {
            current.lastStatus = 'Assistant response stored';
            current.lastError = '';
            saveSettings();
        }
    } catch (error) {
        const current = settings();
        current.lastStatus = `Response persistence failed: ${error.message}`;
        current.lastError = String(error.message || error);
        saveSettings();
        if (current.debug) {
            console.warn('[Atagia Memory] response persistence failed', error);
        }
    }
}

function bindTextInput(selector, key) {
    const current = settings();
    $(selector).on('input', function onInput() {
        current[key] = this.value;
        saveSettings();
    });
}

function bindCheckbox(selector, key) {
    const current = settings();
    $(selector).on('change', function onChange() {
        current[key] = this.checked;
        saveSettings();
    });
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
                        <label for="atagia_memory_user_persona_id">Persona ID</label>
                        <input id="atagia_memory_user_persona_id" type="text" value="${escapeAttribute(current.userPersonaId)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_platform_id">Platform ID</label>
                        <input id="atagia_memory_platform_id" type="text" value="${escapeAttribute(current.platformId)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_character_id">Character ID</label>
                        <input id="atagia_memory_character_id" type="text" value="${escapeAttribute(current.characterId)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_conversation_prefix">Conversation prefix</label>
                        <input id="atagia_memory_conversation_prefix" type="text" value="${escapeAttribute(current.conversationPrefix)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_mode">Mode</label>
                        <input id="atagia_memory_mode" type="text" value="${escapeAttribute(current.mode)}">
                    </div>
                    <div class="atagia-memory-row">
                        <label for="atagia_memory_privacy_mode">Memory privacy mode</label>
                        <select id="atagia_memory_privacy_mode">
                            <option value="balanced" ${current.memoryPrivacyMode === 'balanced' ? 'selected' : ''}>balanced</option>
                            <option value="trusted_private" ${current.memoryPrivacyMode === 'trusted_private' ? 'selected' : ''}>trusted_private</option>
                        </select>
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
                    <pre class="atagia-memory-preview" id="atagia_memory_last_request">${escapeText(current.lastRequest)}</pre>
                </div>
            </div>
        </div>`;
    $('#extensions_settings2').append(html);

    bindCheckbox('#atagia_memory_enabled', 'enabled');
    bindTextInput('#atagia_memory_base_url', 'baseUrl');
    bindTextInput('#atagia_memory_api_key', 'apiKey');
    bindTextInput('#atagia_memory_user_id', 'userId');
    bindTextInput('#atagia_memory_user_persona_id', 'userPersonaId');
    bindTextInput('#atagia_memory_platform_id', 'platformId');
    bindTextInput('#atagia_memory_character_id', 'characterId');
    bindTextInput('#atagia_memory_conversation_prefix', 'conversationPrefix');
    bindTextInput('#atagia_memory_mode', 'mode');
    bindCheckbox('#atagia_memory_debug', 'debug');
    $('#atagia_memory_privacy_mode').on('change', function onChange() {
        current.memoryPrivacyMode = this.value;
        saveSettings();
    });
    $('#atagia_memory_test').on('click', async () => {
        try {
            await atagiaFetch('/v1/models', { method: 'GET' });
            current.lastStatus = 'Atagia connection successful';
            current.lastError = '';
        } catch (error) {
            current.lastStatus = `Atagia connection failed: ${error.message}`;
            current.lastError = String(error.message || error);
        }
        $('#atagia_memory_status').text(current.lastStatus);
        $('#atagia_memory_last_request').text(current.lastRequest);
        saveSettings();
    });

    context.eventSource.on(context.event_types.MESSAGE_RECEIVED, handleMessageReceived);
}

export async function onActivate() {
    const context = getContext();
    context.eventSource.on(context.event_types.APP_READY, renderSettings);
}

export const atagiaInternals = {
    transportId,
    latestUserMessageInfo,
    assistantMessageInfo,
    buildMemoryBlock,
    stableSourceSeq,
    recordAssistantResponse,
};

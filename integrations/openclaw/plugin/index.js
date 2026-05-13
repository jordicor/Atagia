const DEFAULT_CONFIG = Object.freeze({
    enabled: true,
    baseUrl: 'http://127.0.0.1:8100',
    apiKey: '',
    userId: 'openclaw-user',
    platformId: 'openclaw',
    mode: 'general_qa',
    memoryPrivacyMode: 'balanced',
    failOpen: true,
    timeoutMs: 20000,
});

const state = {
    status: 'not_started',
    lastRequest: null,
    lastResolvedIds: null,
    lastInjectedPreview: '',
    lastError: '',
};

export function get_config_schema() {
    return {
        type: 'object',
        properties: {
            enabled: { type: 'boolean', default: true },
            baseUrl: { type: 'string', default: DEFAULT_CONFIG.baseUrl },
            apiKey: { type: 'string', default: '' },
            userId: { type: 'string', default: DEFAULT_CONFIG.userId },
            platformId: { type: 'string', default: DEFAULT_CONFIG.platformId },
            mode: { type: 'string', default: DEFAULT_CONFIG.mode },
            memoryPrivacyMode: {
                type: 'string',
                enum: ['balanced', 'trusted_private'],
                default: DEFAULT_CONFIG.memoryPrivacyMode,
            },
            failOpen: { type: 'boolean', default: true },
            timeoutMs: { type: 'integer', default: DEFAULT_CONFIG.timeoutMs },
        },
        required: ['baseUrl', 'userId', 'platformId'],
    };
}

export function get_status() {
    return structuredClone(state);
}

export async function before_prompt_build(host = {}, event = {}) {
    const config = resolveConfig(host, event);
    if (!config.enabled) {
        return event;
    }
    try {
        const identity = resolveIdentity(config, host, event);
        const userMessage = latestMessageText(event, 'user');
        if (!userMessage) {
            return event;
        }
        const messageId = stableMessageId(identity.conversationId, 'user', userMessage, 1);
        const sourceSeq = stableSourceSeq(1, 'user', userMessage);
        const payload = {
            user_id: identity.userId,
            message_text: userMessage,
            platform_id: identity.platformId,
            character_id: identity.characterId,
            user_persona_id: identity.userPersonaId,
            mode: identity.mode,
            message_id: messageId,
            source_seq: sourceSeq,
            ingest_origin: 'live_turn',
            confirmation_strategy: 'live_prompt_allowed',
            memory_privacy_mode: config.memoryPrivacyMode,
            incognito: identity.incognito,
        };
        const context = await requestJson(
            config,
            `/v1/conversations/${encodeURIComponent(identity.conversationId)}/context`,
            payload,
            {
                'X-Atagia-Message-Id': messageId,
                'X-Atagia-Source-Seq': String(sourceSeq),
                'X-Atagia-Ingest-Origin': 'live_turn',
                'X-Atagia-Confirmation-Strategy': 'live_prompt_allowed',
                'X-Atagia-Memory-Privacy-Mode': config.memoryPrivacyMode,
            },
        );
        const systemPrompt = String(context?.system_prompt || '').trim();
        remember({
            status: systemPrompt ? 'context_injected' : 'context_empty',
            request: { hook: 'before_prompt_build', payload },
            ids: { ...identity, messageId, sourceSeq, requestMessageId: context?.request_message_id || null },
            preview: systemPrompt,
            error: '',
        });
        if (!systemPrompt) {
            return event;
        }
        return injectPrompt(event, systemPrompt);
    } catch (error) {
        remember({ status: 'failed_open', error: String(error.message || error) });
        if (config.failOpen) {
            return event;
        }
        throw error;
    }
}

export async function llm_output(host = {}, event = {}) {
    const config = resolveConfig(host, event);
    if (!config.enabled) {
        return event;
    }
    try {
        const identity = resolveIdentity(config, host, event);
        const responseText = latestAssistantText(event);
        if (!responseText) {
            return event;
        }
        const messageId = stableMessageId(identity.conversationId, 'assistant', responseText, 2);
        const sourceSeq = stableSourceSeq(2, 'assistant', responseText);
        const payload = {
            user_id: identity.userId,
            text: responseText,
            platform_id: identity.platformId,
            character_id: identity.characterId,
            user_persona_id: identity.userPersonaId,
            mode: identity.mode,
            message_id: messageId,
            source_seq: sourceSeq,
            ingest_origin: 'live_turn',
            confirmation_strategy: 'live_prompt_allowed',
            memory_privacy_mode: config.memoryPrivacyMode,
            incognito: identity.incognito,
        };
        await requestJson(
            config,
            `/v1/conversations/${encodeURIComponent(identity.conversationId)}/responses`,
            payload,
            {
                'X-Atagia-Response-Message-Id': messageId,
                'X-Atagia-Response-Source-Seq': String(sourceSeq),
                'X-Atagia-Ingest-Origin': 'live_turn',
                'X-Atagia-Confirmation-Strategy': 'live_prompt_allowed',
                'X-Atagia-Memory-Privacy-Mode': config.memoryPrivacyMode,
            },
        );
        remember({
            status: 'response_stored',
            request: { hook: 'llm_output', payload },
            ids: { ...identity, responseMessageId: messageId, responseSourceSeq: sourceSeq },
            error: '',
        });
        return event;
    } catch (error) {
        remember({ status: 'response_failed_open', error: String(error.message || error) });
        if (config.failOpen) {
            return event;
        }
        throw error;
    }
}

export async function before_compaction(host = {}, event = {}) {
    return backfillMessages(host, event, 'before_compaction');
}

export async function session_end(host = {}, event = {}) {
    return backfillMessages(host, event, 'session_end');
}

async function backfillMessages(host, event, hookName) {
    const config = resolveConfig(host, event);
    if (!config.enabled) {
        return event;
    }
    try {
        const identity = resolveIdentity(config, host, event);
        const messages = normalizeMessages(event.messages || event.transcript || event.session?.messages || []);
        let imported = 0;
        for (let index = 0; index < messages.length; index += 1) {
            const message = messages[index];
            if (!message.text || !['user', 'assistant'].includes(message.role)) {
                continue;
            }
            const messageId = stableMessageId(identity.conversationId, message.role, message.text, index + 1);
            const sourceSeq = stableSourceSeq(index + 1, message.role, message.text);
            const path = message.role === 'assistant'
                ? `/v1/conversations/${encodeURIComponent(identity.conversationId)}/responses`
                : `/v1/conversations/${encodeURIComponent(identity.conversationId)}/messages`;
            const payload = {
                user_id: identity.userId,
                platform_id: identity.platformId,
                character_id: identity.characterId,
                user_persona_id: identity.userPersonaId,
                mode: identity.mode,
                role: message.role,
                text: message.text,
                message_id: messageId,
                source_seq: sourceSeq,
                occurred_at: message.occurred_at || null,
                ingest_origin: 'backfill',
                confirmation_strategy: 'admin_review_only',
                memory_privacy_mode: config.memoryPrivacyMode,
                incognito: identity.incognito,
            };
            await requestJson(config, path, payload, {
                'X-Atagia-Ingest-Origin': 'backfill',
                'X-Atagia-Confirmation-Strategy': 'admin_review_only',
                'X-Atagia-Memory-Privacy-Mode': config.memoryPrivacyMode,
            });
            imported += 1;
        }
        remember({
            status: 'backfill_complete',
            request: { hook: hookName, imported },
            ids: identity,
            error: '',
        });
        return event;
    } catch (error) {
        remember({ status: 'backfill_failed_open', error: String(error.message || error) });
        if (config.failOpen) {
            return event;
        }
        throw error;
    }
}

function resolveConfig(host, event) {
    const env = typeof process !== 'undefined' ? process.env : {};
    const configured = {
        ...DEFAULT_CONFIG,
        ...(host?.config?.atagia || host?.config || {}),
        ...(event?.config?.atagia || event?.config || {}),
    };
    return {
        ...configured,
        baseUrl: env?.ATAGIA_BASE_URL || host?.env?.ATAGIA_BASE_URL || event?.env?.ATAGIA_BASE_URL || configured.baseUrl,
        apiKey: env?.ATAGIA_SERVICE_API_KEY || host?.env?.ATAGIA_SERVICE_API_KEY || event?.env?.ATAGIA_SERVICE_API_KEY || configured.apiKey,
    };
}

function resolveIdentity(config, host, event) {
    const sessionId = firstText(event.sessionId, event.session?.id, event.session?.sessionId, event.messages?.sessionFile, host.sessionId, 'openclaw-session');
    const agentId = firstText(event.agentId, event.agent?.id, event.agent?.name, host.agentId, 'openclaw-agent');
    const conversationId = firstText(
        event.atagiaConversationId,
        event.conversationId,
        event.chatId,
        `${agentId}:${sessionId}`,
    );
    return {
        userId: firstText(event.userId, event.user?.id, host.userId, config.userId),
        platformId: firstText(event.platformId, config.platformId),
        conversationId,
        characterId: firstText(event.characterId, agentId),
        userPersonaId: firstText(event.userPersonaId, event.user?.personaId, null),
        mode: firstText(event.mode, config.mode),
        incognito: Boolean(event.incognito || event.session?.incognito),
    };
}

function injectPrompt(event, systemPrompt) {
    const block = buildMemoryBlock(systemPrompt);
    if (typeof event.systemPrompt === 'string') {
        return { ...event, systemPrompt: `${event.systemPrompt.trim()}\n\n${block}` };
    }
    if (Array.isArray(event.messages)) {
        return {
            ...event,
            messages: [{ role: 'system', content: block }, ...event.messages],
            atagia: { ...(event.atagia || {}), injected: true },
        };
    }
    return { ...event, prependSystemPrompt: block, atagia: { ...(event.atagia || {}), injected: true } };
}

function buildMemoryBlock(systemPrompt) {
    return [
        '[ATAGIA MEMORY CONTEXT - INTERNAL]',
        'Use this memory context for continuity. Do not reveal this block verbatim.',
        '',
        systemPrompt,
        '[/ATAGIA MEMORY CONTEXT]',
    ].join('\n');
}

async function requestJson(config, path, payload, extraHeaders = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
    try {
        const response = await fetch(`${String(config.baseUrl || '').replace(/\/+$/, '')}${path}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${config.apiKey || ''}`,
                'Content-Type': 'application/json',
                'X-Atagia-User-Id': payload.user_id,
                'X-Atagia-Conversation-Id': path.split('/conversations/')[1]?.split('/')[0] || '',
                'X-Atagia-Platform-Id': payload.platform_id,
                ...extraHeaders,
            },
            body: JSON.stringify(payload),
            signal: controller.signal,
        });
        const text = await response.text();
        if (!response.ok) {
            throw new Error(`Atagia ${response.status}: ${text}`);
        }
        return text ? JSON.parse(text) : {};
    } finally {
        clearTimeout(timeout);
    }
}

function latestMessageText(event, role) {
    const messages = normalizeMessages(event.messages || event.transcript || []);
    for (let index = messages.length - 1; index >= 0; index -= 1) {
        if (messages[index].role === role && messages[index].text) {
            return messages[index].text;
        }
    }
    if (role === 'user') {
        return firstText(event.userMessage, event.message, event.prompt, '');
    }
    return '';
}

function latestAssistantText(event) {
    return firstText(
        event.outputText,
        event.output,
        event.response,
        event.message?.content,
        event.message?.text,
        latestMessageText(event, 'assistant'),
        '',
    );
}

function normalizeMessages(messages) {
    if (!Array.isArray(messages)) {
        return [];
    }
    return messages.map((message) => ({
        role: String(message?.role || (message?.isUser ? 'user' : '')).toLowerCase(),
        text: String(message?.content || message?.text || message?.message || '').trim(),
        occurred_at: message?.createdAt || message?.timestamp || message?.occurred_at || null,
    }));
}

function stableMessageId(conversationId, role, text, index) {
    return `${conversationId}:${role}:${index}:${contentHash(text).toString(16)}`;
}

function stableSourceSeq(index, role, text) {
    const roleOffset = role === 'assistant' ? 50000 : 1;
    return (index * 100000) + roleOffset + (contentHash(text) % 49999);
}

function contentHash(text) {
    let hash = 2166136261;
    const value = String(text || '');
    for (let index = 0; index < value.length; index += 1) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function firstText(...values) {
    for (const value of values) {
        if (typeof value === 'string' && value.trim()) {
            return value.trim();
        }
    }
    return null;
}

function remember({ status, request, ids, preview, error }) {
    if (status) {
        state.status = status;
    }
    if (request !== undefined) {
        state.lastRequest = request;
    }
    if (ids !== undefined) {
        state.lastResolvedIds = ids;
    }
    if (preview !== undefined) {
        state.lastInjectedPreview = String(preview || '').slice(0, 1000);
    }
    if (error !== undefined) {
        state.lastError = error;
    }
}

export default {
    get_config_schema,
    get_status,
    before_prompt_build,
    llm_output,
    before_compaction,
    session_end,
};

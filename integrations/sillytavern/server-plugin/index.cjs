'use strict';

const crypto = require('node:crypto');

const PLUGIN_ID = 'atagia-memory';
const PLATFORM_ID = 'sillytavern';
const IDENTITY_SCHEMA = 'atagia.external-message.v1';
const DEFAULT_BASE_URL = 'http://127.0.0.1:8100';
const DEFAULT_TIMEOUT_MS = 20_000;
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
const MAX_TEXT_BYTES = 2 * 1024 * 1024;
const TRANSPORT_ID_PREFIX = '__atagia_b64_';
const SAFE_PATH_ID = /^[A-Za-z0-9_:-][A-Za-z0-9_.:-]*$/;
const FORBIDDEN_BROWSER_FIELDS = new Set([
    'apiKey',
    'api_key',
    'authorization',
    'atagia_message_id',
    'messageId',
    'message_id',
    'platformId',
    'platform_id',
    'serviceKey',
    'service_key',
    'userId',
    'user_id',
]);

const info = Object.freeze({
    id: PLUGIN_ID,
    name: 'Atagia Memory',
    description: 'Keeps Atagia credentials and SillyTavern-to-Atagia user mapping server-side.',
});

class BoundaryError extends Error {
    constructor(statusCode, code) {
        super(code);
        this.name = 'BoundaryError';
        this.statusCode = statusCode;
        this.code = code;
    }
}

class UpstreamError extends Error {
    constructor(statusCode, code) {
        super(code);
        this.name = 'UpstreamError';
        this.statusCode = statusCode;
        this.code = code;
    }
}

function requiredText(value, name) {
    if (typeof value !== 'string' || !value.trim()) {
        throw new Error(`${name} is required`);
    }
    return value.trim();
}

function credentialFingerprint(secret) {
    return `sha256:${crypto.createHash('sha256').update(secret).digest('hex').slice(0, 16)}`;
}

function transportId(value) {
    const text = String(value);
    if (
        text !== '.'
        && text !== '..'
        && !text.startsWith(TRANSPORT_ID_PREFIX)
        && SAFE_PATH_ID.test(text)
    ) {
        return text;
    }
    const encoded = Buffer.from(text, 'utf8')
        .toString('base64')
        .replaceAll('+', '-')
        .replaceAll('/', '_')
        .replace(/=+$/, '');
    return `${TRANSPORT_ID_PREFIX}${encoded}`;
}

function parseBaseUrl(rawValue) {
    const url = new URL(requiredText(rawValue || DEFAULT_BASE_URL, 'ATAGIA_BASE_URL'));
    if (!['http:', 'https:'].includes(url.protocol)) {
        throw new Error('ATAGIA_BASE_URL must use http or https');
    }
    if (url.username || url.password || url.search || url.hash) {
        throw new Error('ATAGIA_BASE_URL must not contain credentials, a query, or a fragment');
    }
    url.pathname = url.pathname.replace(/\/+$/, '');
    return url.toString().replace(/\/+$/, '');
}

function parseUserMap(rawValue) {
    const text = requiredText(rawValue, 'ATAGIA_SILLYTAVERN_USER_MAP');
    let parsed;
    try {
        parsed = JSON.parse(text);
    } catch {
        throw new Error('ATAGIA_SILLYTAVERN_USER_MAP must be a JSON object');
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('ATAGIA_SILLYTAVERN_USER_MAP must be a JSON object');
    }
    const entries = Object.entries(parsed);
    if (entries.length === 0) {
        throw new Error('ATAGIA_SILLYTAVERN_USER_MAP must map at least one user');
    }
    const mapping = new Map();
    for (const [handle, userId] of entries) {
        const normalizedHandle = requiredText(handle, 'SillyTavern user handle');
        const normalizedUserId = requiredText(userId, `Atagia user ID for ${normalizedHandle}`);
        mapping.set(normalizedHandle, normalizedUserId);
    }
    return mapping;
}

function parseTimeoutMs(rawValue) {
    if (rawValue === undefined || rawValue === null || rawValue === '') {
        return DEFAULT_TIMEOUT_MS;
    }
    const value = Number(rawValue);
    if (!Number.isInteger(value) || value < 100 || value > 120_000) {
        throw new Error('ATAGIA_SILLYTAVERN_TIMEOUT_MS must be an integer from 100 to 120000');
    }
    return value;
}

function loadConfig(env = process.env) {
    const serviceKey = requiredText(env.ATAGIA_SERVICE_API_KEY, 'ATAGIA_SERVICE_API_KEY');
    const installationId = requiredText(
        env.ATAGIA_SILLYTAVERN_INSTALLATION_ID,
        'ATAGIA_SILLYTAVERN_INSTALLATION_ID',
    );
    const userMap = parseUserMap(env.ATAGIA_SILLYTAVERN_USER_MAP);
    return Object.freeze({
        baseUrl: parseBaseUrl(env.ATAGIA_BASE_URL),
        installationId,
        serviceKey,
        serviceKeyFingerprint: credentialFingerprint(serviceKey),
        timeoutMs: parseTimeoutMs(env.ATAGIA_SILLYTAVERN_TIMEOUT_MS),
        userMap,
    });
}

function headerValue(request, name) {
    if (typeof request.get === 'function') {
        return request.get(name);
    }
    const headers = request.headers || {};
    return headers[name.toLowerCase()] ?? headers[name];
}

function constantTimeEqual(left, right) {
    if (typeof left !== 'string' || typeof right !== 'string') {
        return false;
    }
    const leftBytes = Buffer.from(left);
    const rightBytes = Buffer.from(right);
    return leftBytes.length === rightBytes.length && crypto.timingSafeEqual(leftBytes, rightBytes);
}

function sameOriginSessionGuard(request, response, next) {
    const handle = request.user?.profile?.handle;
    if (typeof handle !== 'string' || !handle) {
        return response.status(403).json({ error: 'authenticated_session_required' });
    }
    if (!request.session || typeof request.session !== 'object') {
        return response.status(403).json({ error: 'authenticated_session_required' });
    }
    if (request.session.handle && request.session.handle !== handle) {
        return response.status(403).json({ error: 'session_user_mismatch' });
    }
    const fetchSite = headerValue(request, 'sec-fetch-site');
    if (fetchSite && fetchSite !== 'same-origin') {
        return response.status(403).json({ error: 'same_origin_required' });
    }
    const expectedCsrf = request.session.csrfToken;
    const suppliedCsrf = headerValue(request, 'x-csrf-token');
    if (
        expectedCsrf === 'disabled'
        || suppliedCsrf === 'disabled'
        || !constantTimeEqual(expectedCsrf, suppliedCsrf)
    ) {
        return response.status(403).json({ error: 'csrf_validation_failed' });
    }
    return next();
}

function isPlainObject(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        return false;
    }
    const prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null;
}

function rejectBrowserAuthority(body) {
    if (!isPlainObject(body)) {
        throw new BoundaryError(400, 'json_object_required');
    }
    for (const key of Object.keys(body)) {
        if (FORBIDDEN_BROWSER_FIELDS.has(key)) {
            throw new BoundaryError(400, 'server_managed_identity_required');
        }
    }
}

function boundedString(
    body,
    key,
    { required = false, maxLength = 512, maxBytes = null } = {},
) {
    const value = body[key];
    if (value === undefined || value === null || value === '') {
        if (required) {
            throw new BoundaryError(400, `missing_${key}`);
        }
        return null;
    }
    if (typeof value !== 'string') {
        throw new BoundaryError(400, `invalid_${key}`);
    }
    const normalized = value.trim();
    if (
        (required && !normalized)
        || normalized.length > maxLength
        || (maxBytes !== null && Buffer.byteLength(normalized, 'utf8') > maxBytes)
    ) {
        throw new BoundaryError(400, `invalid_${key}`);
    }
    return normalized || null;
}

function sourceSequence(body) {
    const value = body.source_seq;
    if (value === undefined || value === null) {
        return null;
    }
    if (!Number.isSafeInteger(value) || value < 1) {
        throw new BoundaryError(400, 'invalid_source_seq');
    }
    return value;
}

function optionalBoolean(body, key) {
    const value = body[key];
    if (value === undefined || value === null) {
        return null;
    }
    if (typeof value !== 'boolean') {
        throw new BoundaryError(400, `invalid_${key}`);
    }
    return value;
}

function canonicalExternalMessageId({
    installationId,
    hostAccountId,
    mappedUserId,
    hostConversationId,
    sourceNamespace,
    hostMessageId,
    role,
    generationId,
}) {
    const fields = {
        atagia_user_id: mappedUserId,
        generation_id: generationId,
        host_account_id: hostAccountId,
        host_conversation_id: hostConversationId,
        host_installation_id: installationId,
        host_message_id: hostMessageId,
        integration_kind: PLATFORM_ID,
        role,
        schema: IDENTITY_SCHEMA,
        source_namespace: sourceNamespace,
    };
    return `extmsg_${crypto.createHash('sha256').update(JSON.stringify(fields)).digest('hex')}`;
}

function sourceIdentity(body, mappedIdentity, config, role) {
    const sourceSurface = boundedString(body, 'source_surface', {
        required: true,
        maxLength: 32,
    });
    if (sourceSurface !== 'live_event') {
        throw new BoundaryError(400, 'invalid_source_surface');
    }
    const sourceNamespace = boundedString(body, 'source_namespace', {
        required: true,
        maxLength: 32,
    });
    if (!['host_message', 'live_event'].includes(sourceNamespace)) {
        throw new BoundaryError(400, 'invalid_source_namespace');
    }
    const hostConversationId = boundedString(body, 'host_conversation_id', {
        required: true,
        maxLength: 1024,
    });
    const hostMessageId = boundedString(body, 'host_message_id', {
        required: true,
        maxLength: 1024,
    });
    const generationId = boundedString(body, 'host_generation_id', {
        required: true,
        maxLength: 1024,
    });
    return {
        sourceSurface,
        sourceNamespace,
        hostConversationId,
        hostMessageId,
        generationId,
        messageId: canonicalExternalMessageId({
            installationId: config.installationId,
            hostAccountId: mappedIdentity.hostAccountId,
            mappedUserId: mappedIdentity.userId,
            hostConversationId,
            sourceNamespace,
            hostMessageId,
            role,
            generationId,
        }),
    };
}

function commonPayload(body, mappedIdentity, config, role) {
    rejectBrowserAuthority(body);
    const memoryPrivacyMode = boundedString(body, 'memory_privacy_mode', { maxLength: 32 });
    if (memoryPrivacyMode && !['balanced', 'trusted_private'].includes(memoryPrivacyMode)) {
        throw new BoundaryError(400, 'invalid_memory_privacy_mode');
    }
    const identity = sourceIdentity(body, mappedIdentity, config, role);
    return {
        identity,
        payload: {
            user_id: mappedIdentity.userId,
            platform_id: PLATFORM_ID,
            character_id: boundedString(body, 'character_id'),
            user_persona_id: boundedString(body, 'user_persona_id'),
            mode: boundedString(body, 'mode', { maxLength: 64 }),
            message_id: identity.messageId,
            source_seq: sourceSequence(body),
            incognito: optionalBoolean(body, 'incognito'),
            ingest_origin: 'live_turn',
            confirmation_strategy: 'live_prompt_allowed',
            memory_privacy_mode: memoryPrivacyMode,
        },
    };
}

function contextRequest(body, mappedIdentity, config) {
    const common = commonPayload(body, mappedIdentity, config, 'user');
    return {
        conversationId: boundedString(body, 'conversation_id', { required: true, maxLength: 1024 }),
        identity: common.identity,
        payload: {
            ...common.payload,
            message_text: boundedString(body, 'message_text', {
                required: true,
                maxLength: MAX_TEXT_BYTES,
                maxBytes: MAX_TEXT_BYTES,
            }),
        },
    };
}

function responseRequest(body, mappedIdentity, config) {
    const common = commonPayload(body, mappedIdentity, config, 'assistant');
    return {
        conversationId: boundedString(body, 'conversation_id', { required: true, maxLength: 1024 }),
        identity: common.identity,
        payload: {
            ...common.payload,
            text: boundedString(body, 'text', {
                required: true,
                maxLength: MAX_TEXT_BYTES,
                maxBytes: MAX_TEXT_BYTES,
            }),
        },
    };
}

function mappedIdentity(request, config) {
    const handle = request.user.profile.handle;
    const userId = config.userMap.get(handle);
    if (!userId) {
        throw new BoundaryError(403, 'atagia_user_mapping_missing');
    }
    return { hostAccountId: handle, userId };
}

function upstreamHeaders(config, userId, conversationId, payload, operation) {
    const headers = {
        'Authorization': `Bearer ${config.serviceKey}`,
        'Content-Type': 'application/json',
        'X-Atagia-User-Id': userId,
        'X-Atagia-Platform-Id': PLATFORM_ID,
        'X-Atagia-Conversation-Id': conversationId,
        'X-Atagia-Ingest-Origin': 'live_turn',
        'X-Atagia-Confirmation-Strategy': 'live_prompt_allowed',
    };
    if (payload.memory_privacy_mode) {
        headers['X-Atagia-Memory-Privacy-Mode'] = payload.memory_privacy_mode;
    }
    if (operation === 'context') {
        headers['X-Atagia-Message-Id'] = payload.message_id;
    } else {
        headers['X-Atagia-Response-Message-Id'] = payload.message_id;
    }
    if (payload.source_seq !== null) {
        const name = operation === 'context'
            ? 'X-Atagia-Source-Seq'
            : 'X-Atagia-Response-Source-Seq';
        headers[name] = String(payload.source_seq);
    }
    return headers;
}

async function boundedResponseText(response) {
    if (!response.body || typeof response.body.getReader !== 'function') {
        const text = await response.text();
        if (Buffer.byteLength(text, 'utf8') > MAX_RESPONSE_BYTES) {
            throw new UpstreamError(502, 'upstream_response_too_large');
        }
        return text;
    }

    const reader = response.body.getReader();
    const chunks = [];
    let totalBytes = 0;
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            totalBytes += value.byteLength;
            if (totalBytes > MAX_RESPONSE_BYTES) {
                await reader.cancel();
                throw new UpstreamError(502, 'upstream_response_too_large');
            }
            chunks.push(Buffer.from(value.buffer, value.byteOffset, value.byteLength));
        }
    } finally {
        reader.releaseLock();
    }
    return Buffer.concat(chunks, totalBytes).toString('utf8');
}

function safeLogger(logger, level, message) {
    const method = logger?.[level];
    if (typeof method === 'function') {
        method.call(logger, message);
    }
}

function createAtagiaBoundary({ env = process.env, fetchImpl = globalThis.fetch, logger = console } = {}) {
    if (typeof fetchImpl !== 'function') {
        throw new Error('A fetch implementation is required');
    }
    const config = loadConfig(env);
    const activeControllers = new Set();
    let closed = false;

    async function upstream(operation, conversationId, payload, userId) {
        if (closed) {
            throw new UpstreamError(503, 'boundary_closed');
        }
        const controller = new AbortController();
        activeControllers.add(controller);
        const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
        const path = operation === 'health'
            ? '/v1/models'
            : `/v1/conversations/${transportId(conversationId)}/${operation === 'context' ? 'context' : 'responses'}`;
        try {
            const requestOptions = {
                method: operation === 'health' ? 'GET' : 'POST',
                headers: operation === 'health'
                    ? {
                        'Authorization': `Bearer ${config.serviceKey}`,
                        'X-Atagia-User-Id': userId,
                        'X-Atagia-Platform-Id': PLATFORM_ID,
                    }
                    : upstreamHeaders(config, userId, conversationId, payload, operation),
                signal: controller.signal,
            };
            if (operation !== 'health') {
                requestOptions.body = JSON.stringify(payload);
            }
            const response = await fetchImpl(`${config.baseUrl}${path}`, requestOptions);
            const responseText = await boundedResponseText(response);
            if (!response.ok) {
                throw new UpstreamError(502, 'atagia_upstream_rejected');
            }
            return responseText;
        } catch (error) {
            if (error?.name === 'AbortError') {
                throw closed
                    ? new UpstreamError(503, 'boundary_closed')
                    : new UpstreamError(504, 'atagia_upstream_timeout');
            }
            if (error instanceof UpstreamError) {
                throw error;
            }
            throw new UpstreamError(502, 'atagia_upstream_unavailable');
        } finally {
            clearTimeout(timeout);
            activeControllers.delete(controller);
        }
    }

    function route(operation, handler) {
        return async (request, response) => {
            try {
                const identity = mappedIdentity(request, config);
                await handler(request, response, identity);
            } catch (error) {
                const statusCode = error instanceof BoundaryError || error instanceof UpstreamError
                    ? error.statusCode
                    : 500;
                const code = error instanceof BoundaryError || error instanceof UpstreamError
                    ? error.code
                    : 'internal_boundary_error';
                safeLogger(logger, 'warn', `[Atagia Memory] ${operation} failed (${code})`);
                return response.status(statusCode).json({ error: code });
            }
        };
    }

    const health = route('health', async (request, response, identity) => {
        rejectBrowserAuthority(request.body ?? {});
        await upstream('health', null, null, identity.userId);
        return response.status(200).json({ status: 'ok' });
    });

    const context = route('context', async (request, response, identity) => {
        const normalized = contextRequest(request.body, identity, config);
        const text = await upstream(
            'context',
            normalized.conversationId,
            normalized.payload,
            identity.userId,
        );
        let payload;
        try {
            payload = text ? JSON.parse(text) : {};
        } catch {
            throw new UpstreamError(502, 'invalid_atagia_response');
        }
        const upstreamMessageId = typeof payload.request_message_id === 'string'
            ? payload.request_message_id
            : null;
        if (upstreamMessageId && upstreamMessageId !== normalized.payload.message_id) {
            throw new UpstreamError(502, 'atagia_identity_mismatch');
        }
        return response.status(200).json({
            system_prompt: typeof payload.system_prompt === 'string' ? payload.system_prompt : '',
            request_message_id: upstreamMessageId || normalized.payload.message_id,
            message_id: normalized.payload.message_id,
            source_seq: normalized.payload.source_seq,
        });
    });

    const persistResponse = route('response', async (request, response, identity) => {
        const normalized = responseRequest(request.body, identity, config);
        const text = await upstream(
            'response',
            normalized.conversationId,
            normalized.payload,
            identity.userId,
        );
        let payload = {};
        try {
            payload = text ? JSON.parse(text) : {};
        } catch {
            throw new UpstreamError(502, 'invalid_atagia_response');
        }
        if (
            typeof payload.message_id === 'string'
            && payload.message_id !== normalized.payload.message_id
        ) {
            throw new UpstreamError(502, 'atagia_identity_mismatch');
        }
        if (
            Number.isSafeInteger(payload.source_seq)
            && payload.source_seq !== normalized.payload.source_seq
        ) {
            throw new UpstreamError(502, 'atagia_identity_mismatch');
        }
        return response.status(200).json({
            message_id: normalized.payload.message_id,
            source_seq: normalized.payload.source_seq,
        });
    });

    return Object.freeze({
        fingerprint: config.serviceKeyFingerprint,
        mappedUserCount: config.userMap.size,
        register(router) {
            router.use(sameOriginSessionGuard);
            router.post('/health', health);
            router.post('/context', context);
            router.post('/response', persistResponse);
        },
        close() {
            if (closed) {
                return;
            }
            closed = true;
            for (const controller of activeControllers) {
                controller.abort();
            }
            activeControllers.clear();
        },
    });
}

let activeBoundary = null;

async function init(router) {
    if (activeBoundary) {
        activeBoundary.close();
        activeBoundary = null;
    }
    const boundary = createAtagiaBoundary();
    try {
        boundary.register(router);
    } catch (error) {
        boundary.close();
        throw error;
    }
    activeBoundary = boundary;
    console.info(
        `[Atagia Memory] server boundary ready; credential=${boundary.fingerprint}; mapped_users=${boundary.mappedUserCount}`,
    );
}

async function exit() {
    if (!activeBoundary) {
        return;
    }
    activeBoundary.close();
    activeBoundary = null;
}

module.exports = {
    info,
    init,
    exit,
    createAtagiaBoundary,
    credentialFingerprint,
    sameOriginSessionGuard,
    canonicalExternalMessageId,
    transportId,
};

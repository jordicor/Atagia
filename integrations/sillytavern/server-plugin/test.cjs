'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const test = require('node:test');

const plugin = require('./index.cjs');

function secret(label) {
    return `${label}-${crypto.randomBytes(24).toString('base64url')}`;
}

function configEnv(serviceKey, userMap = { alice: 'atagia-alice', bob: 'atagia-bob' }) {
    return {
        ATAGIA_BASE_URL: 'http://atagia.test',
        ATAGIA_SERVICE_API_KEY: serviceKey,
        ATAGIA_SILLYTAVERN_INSTALLATION_ID: 'sillytavern-test-installation',
        ATAGIA_SILLYTAVERN_USER_MAP: JSON.stringify(userMap),
        ATAGIA_SILLYTAVERN_TIMEOUT_MS: '1000',
    };
}

function fakeRouter() {
    const middleware = [];
    const routes = new Map();
    return {
        routes,
        use(handler) {
            middleware.push(handler);
        },
        post(path, ...handlers) {
            routes.set(`POST ${path}`, [...middleware, ...handlers]);
        },
    };
}

async function invoke(router, path, {
    body = {},
    handle = 'alice',
    sessionHandle = handle,
    csrfToken = 'csrf-token',
    suppliedCsrf = csrfToken,
    headers = {},
    includeSession = true,
    includeUser = true,
} = {}) {
    const requestHeaders = {
        'sec-fetch-site': 'same-origin',
        'x-csrf-token': suppliedCsrf,
        ...headers,
    };
    const request = {
        body,
        headers: requestHeaders,
        get(name) {
            return requestHeaders[name.toLowerCase()];
        },
        user: includeUser ? { profile: { handle } } : undefined,
        session: includeSession ? { handle: sessionHandle, csrfToken } : undefined,
    };
    const result = { statusCode: 200, body: undefined };
    const response = {
        status(statusCode) {
            result.statusCode = statusCode;
            return this;
        },
        json(payload) {
            result.body = payload;
            return this;
        },
        sendStatus(statusCode) {
            result.statusCode = statusCode;
            result.body = undefined;
            return this;
        },
    };
    const handlers = router.routes.get(`POST ${path}`);
    assert.ok(handlers, `route ${path} was not registered`);
    let index = 0;
    async function next() {
        const handler = handlers[index++];
        if (!handler) {
            return;
        }
        return handler(request, response, next);
    }
    await next();
    return result;
}

test('exports the pinned SillyTavern server-plugin loader shape', async () => {
    assert.deepEqual(Object.keys(plugin.info).sort(), ['description', 'id', 'name']);
    assert.equal(plugin.info.id, 'atagia-memory');
    assert.equal(typeof plugin.init, 'function');
    assert.equal(typeof plugin.exit, 'function');

    const serviceKey = secret('loader');
    const previous = {
        baseUrl: process.env.ATAGIA_BASE_URL,
        installationId: process.env.ATAGIA_SILLYTAVERN_INSTALLATION_ID,
        serviceKey: process.env.ATAGIA_SERVICE_API_KEY,
        userMap: process.env.ATAGIA_SILLYTAVERN_USER_MAP,
    };
    process.env.ATAGIA_BASE_URL = 'http://atagia.test';
    process.env.ATAGIA_SERVICE_API_KEY = serviceKey;
    process.env.ATAGIA_SILLYTAVERN_INSTALLATION_ID = 'sillytavern-test-installation';
    process.env.ATAGIA_SILLYTAVERN_USER_MAP = JSON.stringify({ alice: 'atagia-alice' });
    const router = fakeRouter();
    const logs = [];
    const originalInfo = console.info;
    console.info = message => logs.push(String(message));
    try {
        await plugin.init(router);
        assert.deepEqual([...router.routes.keys()].sort(), [
            'POST /context',
            'POST /health',
            'POST /response',
        ]);
        assert.equal(logs.length, 1);
        assert.doesNotMatch(logs[0], new RegExp(serviceKey));
        assert.match(logs[0], /sha256:[a-f0-9]{16}/);
    } finally {
        await plugin.exit();
        console.info = originalInfo;
        for (const [name, value] of Object.entries({
            ATAGIA_BASE_URL: previous.baseUrl,
            ATAGIA_SILLYTAVERN_INSTALLATION_ID: previous.installationId,
            ATAGIA_SERVICE_API_KEY: previous.serviceKey,
            ATAGIA_SILLYTAVERN_USER_MAP: previous.userMap,
        })) {
            if (value === undefined) {
                delete process.env[name];
            } else {
                process.env[name] = value;
            }
        }
    }
});

test('failed reconfiguration disables the old boundary without a key fallback', async () => {
    const oldKey = secret('compromised-old-config');
    const previous = {
        baseUrl: process.env.ATAGIA_BASE_URL,
        installationId: process.env.ATAGIA_SILLYTAVERN_INSTALLATION_ID,
        serviceKey: process.env.ATAGIA_SERVICE_API_KEY,
        userMap: process.env.ATAGIA_SILLYTAVERN_USER_MAP,
    };
    process.env.ATAGIA_BASE_URL = 'http://atagia.test';
    process.env.ATAGIA_SERVICE_API_KEY = oldKey;
    process.env.ATAGIA_SILLYTAVERN_INSTALLATION_ID = 'sillytavern-test-installation';
    process.env.ATAGIA_SILLYTAVERN_USER_MAP = JSON.stringify({ alice: 'atagia-alice' });
    const oldRouter = fakeRouter();
    const failedRouter = fakeRouter();
    const logs = [];
    const originalInfo = console.info;
    const originalWarn = console.warn;
    console.info = message => logs.push(String(message));
    console.warn = message => logs.push(String(message));
    try {
        await plugin.init(oldRouter);
        delete process.env.ATAGIA_SERVICE_API_KEY;
        await assert.rejects(
            plugin.init(failedRouter),
            /ATAGIA_SERVICE_API_KEY is required/,
        );
        assert.equal(failedRouter.routes.size, 0);

        const oldRouteResult = await invoke(oldRouter, '/health');
        assert.equal(oldRouteResult.statusCode, 503);
        assert.deepEqual(oldRouteResult.body, { error: 'boundary_closed' });
        assert.equal(logs.some(message => message.includes(oldKey)), false);
    } finally {
        await plugin.exit();
        console.info = originalInfo;
        console.warn = originalWarn;
        for (const [name, value] of Object.entries({
            ATAGIA_BASE_URL: previous.baseUrl,
            ATAGIA_SILLYTAVERN_INSTALLATION_ID: previous.installationId,
            ATAGIA_SERVICE_API_KEY: previous.serviceKey,
            ATAGIA_SILLYTAVERN_USER_MAP: previous.userMap,
        })) {
            if (value === undefined) {
                delete process.env[name];
            } else {
                process.env[name] = value;
            }
        }
    }
});

test('maps each authenticated host user server-side and rejects browser identity claims', async () => {
    const serviceKey = secret('server-only');
    const calls = [];
    const boundary = plugin.createAtagiaBoundary({
        env: configEnv(serviceKey),
        fetchImpl: async (url, options) => {
            calls.push({ url, options });
            return new Response(JSON.stringify({ system_prompt: 'private context' }), { status: 200 });
        },
        logger: { warn() {} },
    });
    const router = fakeRouter();
    boundary.register(router);
    const baseBody = {
        conversation_id: 'conversation-1',
        host_conversation_id: 'host-conversation-1',
        host_message_id: 'message-1',
        host_generation_id: 'default',
        source_namespace: 'host_message',
        source_surface: 'live_event',
        message_text: 'hello',
        source_seq: 1,
        mode: 'companion',
        memory_privacy_mode: 'balanced',
    };

    const alice = await invoke(router, '/context', {
        body: baseBody,
        handle: 'alice',
        headers: { 'x-atagia-user-id': 'attempted-browser-swap' },
    });
    const bob = await invoke(router, '/context', {
        body: {
            ...baseBody,
            conversation_id: 'conversation-2',
            host_conversation_id: 'host-conversation-2',
        },
        handle: 'bob',
    });
    assert.equal(alice.statusCode, 200);
    assert.equal(bob.statusCode, 200);
    assert.equal(calls.length, 2);
    assert.equal(calls[0].options.headers['X-Atagia-User-Id'], 'atagia-alice');
    assert.equal(calls[1].options.headers['X-Atagia-User-Id'], 'atagia-bob');
    assert.equal(JSON.parse(calls[0].options.body).user_id, 'atagia-alice');
    assert.equal(JSON.parse(calls[1].options.body).user_id, 'atagia-bob');
    assert.match(JSON.parse(calls[0].options.body).message_id, /^extmsg_[a-f0-9]{64}$/);
    assert.notEqual(
        JSON.parse(calls[0].options.body).message_id,
        JSON.parse(calls[1].options.body).message_id,
    );
    assert.equal(calls[0].options.headers.Authorization, `Bearer ${serviceKey}`);
    assert.equal(JSON.stringify(alice.body).includes(serviceKey), false);

    const callCount = calls.length;
    const spoofed = await invoke(router, '/context', {
        body: { ...baseBody, user_id: 'victim' },
        handle: 'alice',
    });
    const unmapped = await invoke(router, '/context', {
        body: baseBody,
        handle: 'mallory',
    });
    assert.equal(spoofed.statusCode, 400);
    assert.deepEqual(spoofed.body, { error: 'server_managed_identity_required' });
    assert.equal(unmapped.statusCode, 403);
    assert.deepEqual(unmapped.body, { error: 'atagia_user_mapping_missing' });
    assert.equal(calls.length, callCount);
    boundary.close();
});

test('uses the canonical reversible transport for every dynamic Atagia path', async () => {
    const calls = [];
    const boundary = plugin.createAtagiaBoundary({
        env: configEnv(secret('transport')),
        fetchImpl: async (url, options) => {
            const payload = JSON.parse(options.body);
            calls.push({ url, options, payload });
            return new Response(JSON.stringify({
                message_id: payload.message_id,
                request_message_id: payload.message_id,
                source_seq: payload.source_seq,
            }), { status: 200 });
        },
        logger: { warn() {} },
    });
    const router = fakeRouter();
    boundary.register(router);
    const conversationIds = [
        'ordinary-safe_id:1',
        'id/with/slashes',
        'literal%2Fencoding',
        'id with spaces',
        'unicode-日本語-ñ',
        '__atagia_b64_reserved-prefix',
        '__atagia_b64_aWQvZG91YmxlLWRlY29kZS1ndWFyZA',
    ];

    for (const [index, conversationId] of conversationIds.entries()) {
        const common = {
            conversation_id: conversationId,
            host_conversation_id: `host-${index}`,
            host_message_id: `message-${index}`,
            host_generation_id: 'default',
            source_namespace: 'host_message',
            source_surface: 'live_event',
            source_seq: index + 1,
        };
        const contextResult = await invoke(router, '/context', {
            body: { ...common, message_text: 'question' },
        });
        const responseResult = await invoke(router, '/response', {
            body: { ...common, text: 'answer' },
        });
        assert.equal(contextResult.statusCode, 200);
        assert.equal(responseResult.statusCode, 200);

        const contextCall = calls.at(-2);
        const responseCall = calls.at(-1);
        const encoded = plugin.transportId(conversationId);
        assert.equal(
            new URL(contextCall.url).pathname,
            `/v1/conversations/${encoded}/context`,
        );
        assert.equal(
            new URL(responseCall.url).pathname,
            `/v1/conversations/${encoded}/responses`,
        );
        assert.equal(contextCall.options.headers['X-Atagia-Conversation-Id'], conversationId);
        assert.equal(responseCall.options.headers['X-Atagia-Conversation-Id'], conversationId);
    }

    assert.equal(plugin.transportId('ordinary-safe_id:1'), 'ordinary-safe_id:1');
    assert.notEqual(
        plugin.transportId('__atagia_b64_reserved-prefix'),
        '__atagia_b64_reserved-prefix',
    );
    boundary.close();
});

test('canonical identity is text-free and scoped by every server-side host dimension', () => {
    const base = {
        installationId: 'install-1',
        hostAccountId: 'account-1',
        mappedUserId: 'user-1',
        hostConversationId: 'chat-1',
        sourceNamespace: 'host_message',
        hostMessageId: 'message-1',
        role: 'assistant',
        generationId: 'generation-1',
    };
    const original = plugin.canonicalExternalMessageId(base);
    assert.equal(original, plugin.canonicalExternalMessageId(base));
    for (const [field, value] of [
        ['installationId', 'install-2'],
        ['hostAccountId', 'account-2'],
        ['mappedUserId', 'user-2'],
        ['hostConversationId', 'chat-2'],
        ['sourceNamespace', 'live_event'],
        ['hostMessageId', 'message-2'],
        ['role', 'user'],
        ['generationId', 'generation-2'],
    ]) {
        assert.notEqual(plugin.canonicalExternalMessageId({ ...base, [field]: value }), original);
    }
    assert.equal(Object.hasOwn(base, 'text'), false);
});

test('response route returns the canonical mapping and retries reuse it', async () => {
    const calls = [];
    const boundary = plugin.createAtagiaBoundary({
        env: configEnv(secret('response')),
        fetchImpl: async (_url, options) => {
            const payload = JSON.parse(options.body);
            calls.push(payload);
            return new Response(JSON.stringify({
                message_id: payload.message_id,
                source_seq: payload.source_seq,
            }), { status: 200 });
        },
        logger: { warn() {} },
    });
    const router = fakeRouter();
    boundary.register(router);
    const body = {
        conversation_id: 'atagia-chat',
        host_conversation_id: 'host-chat',
        host_message_id: 'host-message',
        host_generation_id: 'swipe:0',
        source_namespace: 'host_message',
        source_surface: 'live_event',
        text: 'yes',
        source_seq: 2,
    };

    const first = await invoke(router, '/response', { body });
    const retry = await invoke(router, '/response', { body: { ...body, text: 'changed text' } });

    assert.equal(first.statusCode, 200);
    assert.deepEqual(retry.body, first.body);
    assert.match(first.body.message_id, /^extmsg_[a-f0-9]{64}$/);
    assert.equal(first.body.source_seq, 2);
    assert.equal(calls[0].message_id, calls[1].message_id);
    assert.equal(calls[0].source_seq, calls[1].source_seq);
    boundary.close();
});

test('fails closed without the real session and CSRF boundary', async () => {
    const calls = [];
    const boundary = plugin.createAtagiaBoundary({
        env: configEnv(secret('csrf')),
        fetchImpl: async (...args) => {
            calls.push(args);
            return new Response('{}', { status: 200 });
        },
        logger: { warn() {} },
    });
    const router = fakeRouter();
    boundary.register(router);

    const noUser = await invoke(router, '/health', { includeUser: false });
    const noSession = await invoke(router, '/health', { includeSession: false });
    const wrongUser = await invoke(router, '/health', { sessionHandle: 'bob' });
    const noCsrf = await invoke(router, '/health', { suppliedCsrf: '' });
    const disabledCsrf = await invoke(router, '/health', {
        csrfToken: 'disabled',
        suppliedCsrf: 'disabled',
    });
    const crossSite = await invoke(router, '/health', {
        headers: { 'sec-fetch-site': 'cross-site' },
    });

    for (const result of [noUser, noSession, wrongUser, noCsrf, disabledCsrf, crossSite]) {
        assert.equal(result.statusCode, 403);
    }
    assert.equal(calls.length, 0);
    boundary.close();
});

test('returns bounded generic failures and never logs credential or upstream content', async () => {
    const serviceKey = secret('not-for-logs');
    const upstreamPrivateText = secret('upstream-private');
    const logs = [];
    const boundary = plugin.createAtagiaBoundary({
        env: configEnv(serviceKey),
        fetchImpl: async () => new Response(upstreamPrivateText, { status: 401 }),
        logger: { warn(message) { logs.push(String(message)); } },
    });
    const router = fakeRouter();
    boundary.register(router);
    const result = await invoke(router, '/health');

    assert.equal(result.statusCode, 502);
    assert.deepEqual(result.body, { error: 'atagia_upstream_rejected' });
    const visible = JSON.stringify({ result, logs });
    assert.equal(visible.includes(serviceKey), false);
    assert.equal(visible.includes(upstreamPrivateText), false);
    boundary.close();
});

test('enforces byte bounds before forwarding and while streaming upstream content', async () => {
    const serviceKey = secret('bounded');
    const upstreamPrivateText = secret('oversized-upstream');
    const logs = [];
    let fetchCalls = 0;
    const boundary = plugin.createAtagiaBoundary({
        env: configEnv(serviceKey),
        fetchImpl: async () => {
            fetchCalls += 1;
            return new Response(
                `${upstreamPrivateText}${'x'.repeat((2 * 1024 * 1024) + 1)}`,
                { status: 200 },
            );
        },
        logger: { warn(message) { logs.push(String(message)); } },
    });
    const router = fakeRouter();
    boundary.register(router);

    const tooLargeRequest = await invoke(router, '/context', {
        body: {
            conversation_id: 'conversation-1',
            host_conversation_id: 'host-conversation-1',
            host_message_id: 'message-1',
            host_generation_id: 'default',
            source_namespace: 'host_message',
            source_surface: 'live_event',
            message_text: 'é'.repeat(1_100_000),
            source_seq: 1,
        },
    });
    assert.equal(tooLargeRequest.statusCode, 400);
    assert.deepEqual(tooLargeRequest.body, { error: 'invalid_message_text' });
    assert.equal(fetchCalls, 0);

    const tooLargeResponse = await invoke(router, '/health');
    assert.equal(tooLargeResponse.statusCode, 502);
    assert.deepEqual(tooLargeResponse.body, { error: 'upstream_response_too_large' });
    assert.equal(fetchCalls, 1);
    const visible = JSON.stringify({ tooLargeRequest, tooLargeResponse, logs });
    assert.equal(visible.includes(serviceKey), false);
    assert.equal(visible.includes(upstreamPrivateText), false);
    boundary.close();
});

test('keeps the upstream timeout active while reading the response body', async () => {
    const env = configEnv(secret('timeout'));
    env.ATAGIA_SILLYTAVERN_TIMEOUT_MS = '100';
    const boundary = plugin.createAtagiaBoundary({
        env,
        fetchImpl: async (_url, options) => new Response(new ReadableStream({
            start(controller) {
                controller.enqueue(Buffer.from('{'));
                options.signal.addEventListener('abort', () => {
                    controller.error(new DOMException('Aborted', 'AbortError'));
                }, { once: true });
            },
        }), { status: 200 }),
        logger: { warn() {} },
    });
    const router = fakeRouter();
    boundary.register(router);

    const result = await invoke(router, '/health');
    assert.equal(result.statusCode, 504);
    assert.deepEqual(result.body, { error: 'atagia_upstream_timeout' });
    boundary.close();
});

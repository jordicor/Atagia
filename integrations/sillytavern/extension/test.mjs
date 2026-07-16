import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

let importCounter = 0;

function secret(label) {
    return `${label}-${crypto.randomBytes(24).toString('base64url')}`;
}

function storageProbe() {
    const values = new Map();
    const writes = [];
    return {
        writes,
        getItem(key) {
            return values.get(key) ?? null;
        },
        setItem(key, value) {
            writes.push([key, value]);
            values.set(key, String(value));
        },
        removeItem(key) {
            values.delete(key);
        },
        snapshot() {
            return Object.fromEntries(values);
        },
    };
}

function eventSourceProbe() {
    const listeners = new Map();
    return {
        listeners,
        on(name, handler) {
            const handlers = listeners.get(name) || [];
            handlers.push(handler);
            listeners.set(name, handlers);
        },
        removeListener(name, handler) {
            const handlers = listeners.get(name) || [];
            listeners.set(name, handlers.filter(item => item !== handler));
        },
        async emit(name, ...args) {
            for (const handler of listeners.get(name) || []) {
                await handler(...args);
            }
        },
    };
}

function deferred() {
    let resolve;
    const promise = new Promise(resolvePromise => {
        resolve = resolvePromise;
    });
    return { promise, resolve };
}

function installSillyTavernMock(initialSettings = {}) {
    const eventSource = eventSourceProbe();
    const documentListeners = new Map();
    const windowListeners = new Map();
    const localStorage = storageProbe();
    const sessionStorage = storageProbe();
    const context = {
        extensionSettings: {
            atagia_memory: {
                enabled: true,
                userPersonaId: 'persona',
                characterId: 'char',
                conversationPrefix: 'st',
                mode: 'companion',
                memoryPrivacyMode: 'balanced',
                debug: false,
                ...initialSettings,
            },
        },
        chatMetadata: { chat_id: 'chat/with slash' },
        chatId: 'chat/with slash',
        characterId: 'char',
        name1: 'persona',
        chat: [],
        saveMetadataCalls: 0,
        saveChatCalls: 0,
        saveSettingsCalls: 0,
        saveMetadata() {
            this.saveMetadataCalls += 1;
        },
        saveSettingsDebounced() {
            this.saveSettingsCalls += 1;
        },
        async saveChat() {
            this.saveChatCalls += 1;
        },
        getRequestHeaders() {
            return {
                'Content-Type': 'application/json',
                'X-CSRF-Token': 'csrf-token',
            };
        },
        setExtensionPromptCalls: [],
        setExtensionPrompt(...args) {
            this.setExtensionPromptCalls.push(args);
        },
        eventSource,
        eventTypes: {
            MESSAGE_RECEIVED: 'message_received',
            CHAT_CHANGED: 'chat_changed',
            APP_READY: 'app_ready',
        },
    };
    globalThis.SillyTavern = { getContext: () => context };
    globalThis.localStorage = localStorage;
    globalThis.sessionStorage = sessionStorage;
    globalThis.document = {
        addEventListener(name, handler) {
            documentListeners.set(name, handler);
        },
        removeEventListener(name, handler) {
            if (documentListeners.get(name) === handler) {
                documentListeners.delete(name);
            }
        },
    };
    globalThis.addEventListener = (name, handler) => windowListeners.set(name, handler);
    globalThis.removeEventListener = (name, handler) => {
        if (windowListeners.get(name) === handler) {
            windowListeners.delete(name);
        }
    };
    return {
        context,
        documentListeners,
        windowListeners,
        localStorage,
        sessionStorage,
    };
}

async function loadExtension() {
    importCounter += 1;
    return import(`./index.js?test=${importCounter}`);
}

async function exerciseLateStaleContextCompletion(lateResponse) {
    const { context } = installSillyTavernMock();
    const fetchStarted = deferred();
    const releaseFetch = deferred();
    const calls = [];
    globalThis.fetch = async (_url, options) => {
        const payload = JSON.parse(options.body);
        calls.push(payload);
        if (payload.host_conversation_id !== 'chat-B') {
            fetchStarted.resolve();
            return releaseFetch.promise;
        }
        return new Response(JSON.stringify({
            system_prompt: 'Current B context',
            request_message_id: 'mapped-B',
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const chatA = [{ is_user: true, mes: 'question A', extra: {} }];
    context.chat = chatA;
    const coreChatA = chatA.map((entry, index) => ({ ...entry, index }));

    const slowA = globalThis.atagiaMemoryInterceptor(coreChatA, 0, null, 'normal');
    await fetchStarted.promise;
    await context.eventSource.emit(context.eventTypes.CHAT_CHANGED, 'chat-B');
    context.chatId = 'chat-B';
    context.chatMetadata = { chat_id: 'chat-B' };
    const chatB = [{ is_user: true, mes: 'question B', extra: {} }];
    context.chat = chatB;
    const coreChatB = chatB.map((entry, index) => ({ ...entry, index }));
    await globalThis.atagiaMemoryInterceptor(coreChatB, 0, null, 'normal');
    assert.match(context.setExtensionPromptCalls.at(-1)[1], /Current B context/);

    releaseFetch.resolve(lateResponse());
    await slowA;

    assert.equal(calls.length, 2);
    assert.match(context.setExtensionPromptCalls.at(-1)[1], /Current B context/);
    assert.equal(
        Object.hasOwn(chatA[0].extra.atagia_source_identity, 'atagia_message_id'),
        false,
    );
    assert.equal(chatB[0].extra.atagia_source_identity.atagia_message_id, 'mapped-B');
    assert.equal(module.atagiaInternals.runtimeSnapshot().status, 'Context injected');
    module.atagiaInternals.dispose();
}

async function exerciseSameSourceLateContextCompletion(lateResponse) {
    const { context } = installSillyTavernMock({ debug: true });
    const fetchStarted = deferred();
    const releaseFetch = deferred();
    let callCount = 0;
    globalThis.fetch = async () => {
        callCount += 1;
        if (callCount === 1) {
            fetchStarted.resolve();
            return releaseFetch.promise;
        }
        return new Response(JSON.stringify({
            system_prompt: 'Current context',
            request_message_id: 'mapped-current',
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const user = { is_user: true, mes: 'same user', extra: {} };
    context.chat = [user];
    const coreChat = [{ ...user, index: 0 }];

    const slow = globalThis.atagiaMemoryInterceptor(coreChat, 0, null, 'normal');
    await fetchStarted.promise;
    await globalThis.atagiaMemoryInterceptor(coreChat, 0, null, 'normal');
    assert.match(context.setExtensionPromptCalls.at(-1)[1], /Current context/);
    const promptCallCount = context.setExtensionPromptCalls.length;
    const snapshot = module.atagiaInternals.runtimeSnapshot();

    releaseFetch.resolve(lateResponse());
    await slow;

    assert.equal(callCount, 2);
    assert.equal(user.extra.atagia_source_identity.atagia_message_id, 'mapped-current');
    assert.equal(context.setExtensionPromptCalls.length, promptCallCount);
    assert.match(context.setExtensionPromptCalls.at(-1)[1], /Current context/);
    assert.deepEqual(module.atagiaInternals.runtimeSnapshot(), snapshot);
    module.atagiaInternals.dispose();
}

async function exerciseLateAssistantGenerationCompletion(lateResponse) {
    const { context } = installSillyTavernMock({ debug: true });
    const fetchStarted = deferred();
    const releaseFetch = deferred();
    let callCount = 0;
    globalThis.fetch = async () => {
        callCount += 1;
        if (callCount === 1) {
            fetchStarted.resolve();
            return releaseFetch.promise;
        }
        return new Response(JSON.stringify({
            message_id: 'mapped-new-generation',
            source_seq: null,
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const assistant = {
        is_user: false,
        mes: 'old generation',
        swipe_id: 0,
        swipe_info: [{ extra: { gen_id: 'generation-0' } }],
        extra: {},
    };
    context.chat = [
        { is_user: true, mes: 'question', extra: {} },
        assistant,
    ];

    const slowOld = context.eventSource.emit(
        context.eventTypes.MESSAGE_RECEIVED,
        1,
        'normal',
    );
    await fetchStarted.promise;
    assistant.mes = 'new generation';
    assistant.swipe_id = 1;
    assistant.swipe_info.push({ extra: { gen_id: 'generation-1' } });
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'regenerate');
    const snapshot = module.atagiaInternals.runtimeSnapshot();

    releaseFetch.resolve(lateResponse());
    await slowOld;

    const identities = assistant.extra.atagia_source_identities;
    const oldIdentity = identities.find(item => item.host_generation_id === 'host:generation-0');
    const newIdentity = identities.find(item => item.host_generation_id === 'host:generation-1');
    assert.equal(callCount, 2);
    assert.equal(Object.hasOwn(oldIdentity, 'atagia_message_id'), false);
    assert.equal(newIdentity.atagia_message_id, 'mapped-new-generation');
    assert.equal(assistant.extra.atagia_source_identity, newIdentity);
    assert.deepEqual(module.atagiaInternals.runtimeSnapshot(), snapshot);
    module.atagiaInternals.dispose();
}

function assertBrowserRequestHasNoServerAuthority(call) {
    const visible = JSON.stringify(call);
    assert.doesNotMatch(visible, /authorization/i);
    assert.doesNotMatch(visible, /x-atagia-user-id/i);
    const payload = JSON.parse(call.options.body);
    assert.equal(Object.hasOwn(payload, 'user_id'), false);
    assert.equal(Object.hasOwn(payload, 'userId'), false);
    assert.equal(Object.hasOwn(payload, 'platform_id'), false);
    assert.deepEqual(call.options.headers, {
        'Content-Type': 'application/json',
        'X-CSRF-Token': 'csrf-token',
    });
    assert.equal(call.options.credentials, 'same-origin');
}

test('interceptor uses the same-origin server route without browser credentials', async () => {
    const { context } = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response(JSON.stringify({
            system_prompt: 'Remember this preference.',
            request_message_id: 'stored-user-1',
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    context.setExtensionPromptCalls.length = 0;
    const chat = [{ is_user: true, mes: 'Hello', send_date: '2026-01-01T00:00:00Z' }];
    context.chat = chat;

    await globalThis.atagiaMemoryInterceptor(chat, 0, null, 'normal');

    assert.equal(chat.length, 1);
    assert.equal(context.setExtensionPromptCalls.length, 1);
    assert.match(context.setExtensionPromptCalls[0][1], /ATAGIA MEMORY CONTEXT/);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/plugins/atagia-memory/context');
    assertBrowserRequestHasNoServerAuthority(calls[0]);
    const payload = JSON.parse(calls[0].options.body);
    assert.equal(payload.message_text, 'Hello');
    assert.equal(Object.hasOwn(payload, 'message_id'), false);
    assert.equal(payload.host_conversation_id, 'chat/with slash');
    assert.match(payload.host_message_id, /^stmsg_/);
    assert.equal(payload.host_generation_id, 'default');
    assert.equal(payload.source_namespace, 'host_message');
    assert.equal(payload.source_surface, 'live_event');
    assert.equal(typeof payload.source_seq, 'number');
    assert.equal(chat[0].extra.atagia_source_identity.atagia_message_id, 'stored-user-1');
    assert.ok(context.saveChatCalls >= 2);
    module.atagiaInternals.dispose();
});

test('pinned 1.18.0 cloned coreChat resolves and mutates the canonical user entry', async () => {
    const { context } = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response(JSON.stringify({
            system_prompt: 'Canonical memory.',
            request_message_id: 'stored-cloned-user',
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const canonicalUser = {
        is_user: true,
        mes: 'Canonical raw user text',
        extra: {},
    };
    context.chat = [
        { is_system: true, is_user: false, mes: 'filtered system row', extra: {} },
        canonicalUser,
    ];
    const coreChat = context.chat
        .filter(entry => !entry.is_system)
        .map((entry, index) => ({
            ...entry,
            mes: 'Regex-transformed prompt text',
            index,
        }));

    await globalThis.atagiaMemoryInterceptor(coreChat, 0, null, 'normal');

    assert.equal(calls.length, 1);
    const payload = JSON.parse(calls[0].options.body);
    assert.equal(payload.message_text, 'Canonical raw user text');
    assert.equal(canonicalUser.extra.atagia_source_identity.atagia_message_id, 'stored-cloned-user');
    assert.equal(coreChat[0].mes, 'Regex-transformed prompt text');
    assert.equal(coreChat[0].extra, canonicalUser.extra);
    assert.match(context.setExtensionPromptCalls.at(-1)[1], /Canonical memory/);
    module.atagiaInternals.dispose();
});

test('a stale cloned coreChat cannot bind to the newly active canonical chat', async () => {
    const { context } = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (...args) => {
        calls.push(args);
        return new Response(JSON.stringify({ system_prompt: 'must not inject' }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const chatA = [{ is_user: true, mes: 'question A', extra: {} }];
    context.chat = chatA;
    const staleCoreChat = chatA.map((entry, index) => ({ ...entry, index }));
    await context.eventSource.emit(context.eventTypes.CHAT_CHANGED, 'chat-B');
    context.chatId = 'chat-B';
    context.chatMetadata = { chat_id: 'chat-B' };
    context.chat = [{ is_user: true, mes: 'question B', extra: {} }];
    context.setExtensionPromptCalls.length = 0;

    await globalThis.atagiaMemoryInterceptor(staleCoreChat, 0, null, 'normal');

    assert.equal(calls.length, 0);
    assert.equal(Object.hasOwn(chatA[0].extra, 'atagia_source_identity'), false);
    assert.equal(Object.hasOwn(context.chat[0].extra, 'atagia_source_identity'), false);
    assert.equal(
        context.setExtensionPromptCalls.some(call => String(call[1]).includes('must not inject')),
        false,
    );
    module.atagiaInternals.dispose();
});

test('a current interceptor chat with no user clears its own stale prompt', async () => {
    const { context } = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (...args) => {
        calls.push(args);
        return new Response('{}', { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    context.chat = [{ is_user: false, mes: 'assistant only', extra: {} }];
    context.setExtensionPrompt('atagia_memory', 'same-chat stale prompt', 0, 1, false);

    await globalThis.atagiaMemoryInterceptor(
        context.chat.map((entry, index) => ({ ...entry, index })),
        0,
        null,
        'normal',
    );

    assert.equal(calls.length, 0);
    assert.equal(context.setExtensionPromptCalls.at(-1)[1], '');
    assert.equal(module.atagiaInternals.runtimeSnapshot().status, 'No Atagia context returned');
    module.atagiaInternals.dispose();
});

test('only the canonical empty chat can clear an empty-chat prompt', async () => {
    const { context } = installSillyTavernMock();
    globalThis.fetch = async () => {
        throw new Error('empty chat must not fetch context');
    };
    const module = await loadExtension();
    await module.onActivate();
    context.chat = [];
    context.setExtensionPrompt('atagia_memory', 'current empty-chat prompt', 0, 1, false);
    const initialCalls = context.setExtensionPromptCalls.length;

    await globalThis.atagiaMemoryInterceptor([], 0, null, 'normal');
    assert.equal(context.setExtensionPromptCalls.length, initialCalls);

    await globalThis.atagiaMemoryInterceptor(context.chat, 0, null, 'normal');
    assert.equal(context.setExtensionPromptCalls.at(-1)[1], '');
    assert.equal(module.atagiaInternals.runtimeSnapshot().status, 'No Atagia context returned');
    module.atagiaInternals.dispose();
});

test('empty-chat completion rechecks that no user appeared at the microtask boundary', async () => {
    const { context } = installSillyTavernMock();
    globalThis.fetch = async () => {
        throw new Error('empty chat must not fetch context');
    };
    const module = await loadExtension();
    await module.onActivate();
    context.chat = [];
    context.setExtensionPrompt('atagia_memory', 'prompt owned by the next user', 0, 1, false);
    const promptCallCount = context.setExtensionPromptCalls.length;

    const interception = globalThis.atagiaMemoryInterceptor(context.chat, 0, null, 'normal');
    context.chat.push({ is_user: true, mes: 'new user', extra: {} });
    await interception;

    assert.equal(context.setExtensionPromptCalls.length, promptCallCount);
    assert.equal(context.setExtensionPromptCalls.at(-1)[1], 'prompt owned by the next user');
    module.atagiaInternals.dispose();
});

test('a shared-prefix stale clone cannot claim or overwrite the newer user operation', async () => {
    const { context } = installSillyTavernMock({ debug: true });
    const calls = [];
    globalThis.fetch = async (_url, options) => {
        calls.push(JSON.parse(options.body));
        return new Response(JSON.stringify({
            system_prompt: 'Newer user context',
            request_message_id: 'mapped-newer-user',
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const commonUser = { is_user: true, mes: 'common ancestor', extra: {} };
    const newerUser = { is_user: true, mes: 'newer user', extra: {} };
    context.chat = [
        commonUser,
        { is_user: false, mes: 'assistant between users', extra: {} },
        newerUser,
    ];
    const stalePrefix = [{ ...commonUser, index: 0 }];
    const currentCoreChat = context.chat.map((entry, index) => ({ ...entry, index }));

    await globalThis.atagiaMemoryInterceptor(currentCoreChat, 0, null, 'normal');
    const promptCallCount = context.setExtensionPromptCalls.length;
    const snapshot = module.atagiaInternals.runtimeSnapshot();
    await globalThis.atagiaMemoryInterceptor(stalePrefix, 0, null, 'normal');

    assert.equal(calls.length, 1);
    assert.equal(Object.hasOwn(commonUser.extra, 'atagia_source_identity'), false);
    assert.equal(newerUser.extra.atagia_source_identity.atagia_message_id, 'mapped-newer-user');
    assert.equal(context.setExtensionPromptCalls.length, promptCallCount);
    assert.match(context.setExtensionPromptCalls.at(-1)[1], /Newer user context/);
    assert.deepEqual(module.atagiaInternals.runtimeSnapshot(), snapshot);

    module.atagiaInternals.dispose();
});

test('stale assistant-only and quiet callbacks cannot touch the active prompt', async () => {
    const { context } = installSillyTavernMock({ debug: true });
    globalThis.fetch = async () => new Response(JSON.stringify({
        system_prompt: 'Active B context',
        request_message_id: 'mapped-B',
    }), { status: 200 });
    const module = await loadExtension();
    await module.onActivate();
    const chatA = [{ is_user: false, mes: 'assistant A', extra: {} }];
    const staleAssistantClone = chatA.map((entry, index) => ({ ...entry, index }));
    context.chat = chatA;
    await context.eventSource.emit(context.eventTypes.CHAT_CHANGED, 'chat-B');
    context.chatId = 'chat-B';
    context.chatMetadata = { chat_id: 'chat-B' };
    context.chat = [{ is_user: true, mes: 'question B', extra: {} }];
    const currentCoreChat = context.chat.map((entry, index) => ({ ...entry, index }));
    await globalThis.atagiaMemoryInterceptor(currentCoreChat, 0, null, 'normal');
    const promptCallCount = context.setExtensionPromptCalls.length;
    const snapshot = module.atagiaInternals.runtimeSnapshot();

    await globalThis.atagiaMemoryInterceptor(staleAssistantClone, 0, null, 'normal');
    await globalThis.atagiaMemoryInterceptor(staleAssistantClone, 0, null, 'quiet');

    assert.equal(context.setExtensionPromptCalls.length, promptCallCount);
    assert.match(context.setExtensionPromptCalls.at(-1)[1], /Active B context/);
    assert.deepEqual(module.atagiaInternals.runtimeSnapshot(), snapshot);

    await globalThis.atagiaMemoryInterceptor(currentCoreChat, 0, null, 'quiet');
    assert.equal(context.setExtensionPromptCalls.at(-1)[1], '');
    module.atagiaInternals.dispose();
});

test('canonical clone provenance indexing stays linear in chat size', async () => {
    const { context } = installSillyTavernMock();
    const module = await loadExtension();
    await module.onActivate();
    let extraReads = 0;
    context.chat = Array.from({ length: 2_000 }, (_, index) => {
        const extra = {};
        const entry = {
            is_user: index % 2 === 0,
            mes: `message ${index}`,
        };
        Object.defineProperty(entry, 'extra', {
            enumerable: true,
            get() {
                extraReads += 1;
                return extra;
            },
        });
        return entry;
    });
    const coreChat = context.chat.map((entry, index) => ({ ...entry, index }));
    extraReads = 0;

    const resolved = module.atagiaInternals.resolveInterceptorChat(coreChat);

    assert.equal(resolved?.entry, context.chat[1_998]);
    assert.ok(extraReads <= context.chat.length + 5, `extra reads were ${extraReads}`);
    module.atagiaInternals.dispose();
});

test('a late successful context request cannot erase the new chat prompt', async () => {
    await exerciseLateStaleContextCompletion(() => new Response(JSON.stringify({
        system_prompt: 'Stale A context',
        request_message_id: 'mapped-A',
    }), { status: 200 }));
});

test('a late failed context request cannot erase the new chat prompt', async () => {
    await exerciseLateStaleContextCompletion(() => new Response('', { status: 503 }));
});

test('a newer operation for the same user wins over an older success', async () => {
    await exerciseSameSourceLateContextCompletion(() => new Response(JSON.stringify({
        system_prompt: 'Older context',
        request_message_id: 'mapped-older',
    }), { status: 200 }));
});

test('a newer operation for the same user wins over an older failure', async () => {
    await exerciseSameSourceLateContextCompletion(() => new Response('', { status: 503 }));
});

test('a newly appended user invalidates a deferred operation in the same chat array', async () => {
    const { context } = installSillyTavernMock({ debug: true });
    const fetchStarted = deferred();
    const releaseFetch = deferred();
    let callCount = 0;
    globalThis.fetch = async () => {
        callCount += 1;
        if (callCount === 1) {
            fetchStarted.resolve();
            return releaseFetch.promise;
        }
        return new Response(JSON.stringify({
            system_prompt: 'New user context',
            request_message_id: 'mapped-new-user',
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const firstUser = { is_user: true, mes: 'first user', extra: {} };
    context.chat = [firstUser];
    const firstCoreChat = [{ ...firstUser, index: 0 }];

    const slowFirst = globalThis.atagiaMemoryInterceptor(
        firstCoreChat,
        0,
        null,
        'normal',
    );
    await fetchStarted.promise;
    const newUser = { is_user: true, mes: 'new user', extra: {} };
    context.chat.push(newUser);
    const currentCoreChat = context.chat.map((entry, index) => ({ ...entry, index }));
    await globalThis.atagiaMemoryInterceptor(currentCoreChat, 0, null, 'normal');
    const promptCallCount = context.setExtensionPromptCalls.length;
    const snapshot = module.atagiaInternals.runtimeSnapshot();

    releaseFetch.resolve(new Response(JSON.stringify({
        system_prompt: 'Old user context',
        request_message_id: 'mapped-old-user',
    }), { status: 200 }));
    await slowFirst;

    assert.equal(callCount, 2);
    assert.equal(
        Object.hasOwn(firstUser.extra.atagia_source_identity, 'atagia_message_id'),
        false,
    );
    assert.equal(newUser.extra.atagia_source_identity.atagia_message_id, 'mapped-new-user');
    assert.equal(context.setExtensionPromptCalls.length, promptCallCount);
    assert.match(context.setExtensionPromptCalls.at(-1)[1], /New user context/);
    assert.deepEqual(module.atagiaInternals.runtimeSnapshot(), snapshot);
    module.atagiaInternals.dispose();
});

test('disabling memory clears promptly and invalidates an in-flight context request', async () => {
    const { context } = installSillyTavernMock({ debug: true });
    const fetchStarted = deferred();
    const releaseFetch = deferred();
    globalThis.fetch = async () => {
        fetchStarted.resolve();
        return releaseFetch.promise;
    };
    const module = await loadExtension();
    await module.onActivate();
    const user = { is_user: true, mes: 'in-flight user', extra: {} };
    context.chat = [user];
    context.setExtensionPrompt('atagia_memory', 'old prompt', 0, 1, false);
    const slow = globalThis.atagiaMemoryInterceptor(context.chat, 0, null, 'normal');
    await fetchStarted.promise;

    module.atagiaInternals.setMemoryEnabled(false);
    assert.equal(context.extensionSettings.atagia_memory.enabled, false);
    assert.equal(context.setExtensionPromptCalls.at(-1)[1], '');
    assert.equal(module.atagiaInternals.runtimeSnapshot().status, 'Atagia memory disabled');
    const promptCallCount = context.setExtensionPromptCalls.length;
    const snapshot = module.atagiaInternals.runtimeSnapshot();

    releaseFetch.resolve(new Response(JSON.stringify({
        system_prompt: 'must not inject',
        request_message_id: 'must-not-map',
    }), { status: 200 }));
    await slow;

    assert.equal(
        Object.hasOwn(user.extra.atagia_source_identity, 'atagia_message_id'),
        false,
    );
    assert.equal(context.setExtensionPromptCalls.length, promptCallCount);
    assert.deepEqual(module.atagiaInternals.runtimeSnapshot(), snapshot);
    module.atagiaInternals.setMemoryEnabled(true);
    assert.equal(
        module.atagiaInternals.runtimeSnapshot().status,
        'Atagia memory enabled; awaiting next context',
    );
    module.atagiaInternals.dispose();
});

test('a late old assistant generation cannot replace a newer successful mapping', async () => {
    await exerciseLateAssistantGenerationCompletion(() => new Response(JSON.stringify({
        message_id: 'mapped-old-generation',
        source_seq: 2,
    }), { status: 200 }));
});

test('a late old assistant failure cannot alter newer success diagnostics', async () => {
    await exerciseLateAssistantGenerationCompletion(() => new Response('', { status: 503 }));
});

test('disable during assistant identity persistence prevents dispatch and mapping', async () => {
    const { context } = installSillyTavernMock({ debug: true });
    const saveStarted = deferred();
    const releaseSave = deferred();
    const calls = [];
    context.saveChat = async function saveChat() {
        this.saveChatCalls += 1;
        saveStarted.resolve();
        await releaseSave.promise;
    };
    globalThis.fetch = async (...args) => {
        calls.push(args);
        return new Response(JSON.stringify({ message_id: 'must-not-map' }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const assistant = { is_user: false, mes: 'assistant', swipe_id: 0, extra: {} };
    context.chat = [
        { is_user: true, mes: 'question', extra: {} },
        assistant,
    ];

    const response = context.eventSource.emit(
        context.eventTypes.MESSAGE_RECEIVED,
        1,
        'normal',
    );
    await saveStarted.promise;
    module.atagiaInternals.setMemoryEnabled(false);
    const snapshot = module.atagiaInternals.runtimeSnapshot();
    releaseSave.resolve();
    await response;

    assert.equal(calls.length, 0);
    assert.equal(context.saveChatCalls, 1);
    assert.equal(
        Object.hasOwn(assistant.extra.atagia_source_identity, 'atagia_message_id'),
        false,
    );
    assert.deepEqual(module.atagiaInternals.runtimeSnapshot(), snapshot);
    module.atagiaInternals.dispose();
});

test('disable after assistant dispatch prevents late mapping and UI effects', async () => {
    const { context } = installSillyTavernMock({ debug: true });
    const fetchStarted = deferred();
    const releaseFetch = deferred();
    globalThis.fetch = async () => {
        fetchStarted.resolve();
        return releaseFetch.promise;
    };
    const module = await loadExtension();
    await module.onActivate();
    const assistant = { is_user: false, mes: 'assistant', swipe_id: 0, extra: {} };
    context.chat = [
        { is_user: true, mes: 'question', extra: {} },
        assistant,
    ];

    const response = context.eventSource.emit(
        context.eventTypes.MESSAGE_RECEIVED,
        1,
        'normal',
    );
    await fetchStarted.promise;
    module.atagiaInternals.setMemoryEnabled(false);
    const snapshot = module.atagiaInternals.runtimeSnapshot();
    const saveCount = context.saveChatCalls;
    releaseFetch.resolve(new Response(JSON.stringify({
        message_id: 'must-not-map',
    }), { status: 200 }));
    await response;

    assert.equal(context.saveChatCalls, saveCount);
    assert.equal(
        Object.hasOwn(assistant.extra.atagia_source_identity, 'atagia_message_id'),
        false,
    );
    assert.deepEqual(module.atagiaInternals.runtimeSnapshot(), snapshot);
    module.atagiaInternals.dispose();
});

test('real MESSAGE_RECEIVED primitive resolves the host entry and persists once', async () => {
    const { context } = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        const payload = JSON.parse(options.body);
        return new Response(JSON.stringify({
            message_id: 'stored-assistant-1',
            source_seq: payload.source_seq,
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    context.chat = [
        { is_user: true, mes: 'Hello', send_date: '2026-01-01T00:00:00Z' },
        { is_user: false, mes: 'Hi there.', send_date: '2026-01-01T00:00:01Z', swipe_id: 'a' },
    ];

    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/plugins/atagia-memory/response');
    assertBrowserRequestHasNoServerAuthority(calls[0]);
    const payload = JSON.parse(calls[0].options.body);
    assert.equal(payload.text, 'Hi there.');
    assert.equal(Object.hasOwn(payload, 'message_id'), false);
    assert.match(payload.host_message_id, /^stmsg_/);
    assert.equal(payload.host_generation_id, 'swipe:0');
    assert.equal(typeof payload.source_seq, 'number');
    assert.equal(
        context.chat[1].extra.atagia_source_identity.atagia_message_id,
        'stored-assistant-1',
    );
    module.atagiaInternals.dispose();
});

test('retry, repeated text, swipe regeneration, and reload retain host-native identity', async () => {
    const { context } = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (url, options) => {
        const payload = JSON.parse(options.body);
        calls.push({ url, payload });
        return new Response(JSON.stringify({
            message_id: `mapped:${payload.host_message_id}:${payload.host_generation_id}`,
            source_seq: payload.source_seq,
        }), { status: 200 });
    };
    const firstModule = await loadExtension();
    await firstModule.onActivate();
    context.chat = [
        { is_user: true, mes: 'question' },
        {
            is_user: false,
            mes: 'yes',
            swipe_id: 0,
            swipe_info: [{ extra: { gen_id: 'generation-0' } }],
        },
    ];

    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');
    assert.equal(calls.length, 1, 'a server-confirmed host generation is not resent');

    const originalHostMessageId = calls[0].payload.host_message_id;
    context.chat[1].mes = 'regenerated answer';
    context.chat[1].swipe_id = 1;
    context.chat[1].swipe_info.push({ extra: { gen_id: 'generation-1' } });
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'regenerate');
    assert.equal(calls[1].payload.host_message_id, originalHostMessageId);
    assert.notEqual(calls[1].payload.host_generation_id, calls[0].payload.host_generation_id);
    assert.equal(calls[1].payload.source_seq, null);
    assert.equal(context.chat[1].extra.atagia_source_identities.length, 2);
    assert.equal(
        context.chat[1].extra.atagia_source_identity.host_generation_id,
        'host:generation-1',
    );

    context.chat.push(
        { is_user: true, mes: 'another question' },
        { is_user: false, mes: 'yes', swipe_id: 0 },
    );
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 3, 'normal');
    assert.notEqual(calls[2].payload.host_message_id, originalHostMessageId);
    assert.equal(calls[2].payload.source_seq, 4);

    const selectedPayload = calls[1].payload;
    context.chat.splice(2);
    const secondModule = await loadExtension();
    await secondModule.onActivate();
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'regenerate');
    assert.equal(calls.length, 3, 'reload reuses the persisted server confirmation');
    assert.equal(
        context.chat[1].extra.atagia_source_identity.host_message_id,
        selectedPayload.host_message_id,
    );
    assert.equal(
        context.chat[1].extra.atagia_source_identity.host_generation_id,
        selectedPayload.host_generation_id,
    );
    assert.equal(
        context.chat[1].extra.atagia_source_identity.source_seq,
        selectedPayload.source_seq,
    );
    secondModule.atagiaInternals.dispose();
});

test('a failed assistant write remains retryable and stops after confirmation', async () => {
    const { context } = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (_url, options) => {
        const payload = JSON.parse(options.body);
        calls.push(payload);
        if (calls.length === 1) {
            return new Response('', { status: 503 });
        }
        return new Response(JSON.stringify({
            message_id: `mapped:${payload.host_message_id}:${payload.host_generation_id}`,
            source_seq: payload.source_seq,
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    context.chat = [
        { is_user: true, mes: 'question' },
        { is_user: false, mes: 'answer', swipe_id: 0 },
    ];

    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');
    assert.equal(
        Object.hasOwn(context.chat[1].extra.atagia_source_identity, 'atagia_message_id'),
        false,
    );
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');

    assert.equal(calls.length, 2);
    assert.equal(calls[0].host_message_id, calls[1].host_message_id);
    assert.equal(calls[0].host_generation_id, calls[1].host_generation_id);
    assert.match(
        context.chat[1].extra.atagia_source_identity.atagia_message_id,
        /^mapped:/,
    );
    module.atagiaInternals.dispose();
});

test('chat switch scopes the same host-local ID to the active host conversation', async () => {
    const { context } = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (_url, options) => {
        const payload = JSON.parse(options.body);
        calls.push(payload);
        return new Response(JSON.stringify({
            message_id: `mapped-${calls.length}`,
            source_seq: payload.source_seq,
        }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    context.chat = [
        { is_user: true, mes: 'question' },
        {
            is_user: false,
            mes: 'same answer',
            swipe_id: 0,
            extra: { atagia_host_message_id: 'same-local-message-id' },
        },
    ];
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');

    context.setExtensionPrompt('atagia_memory', 'stale prompt', 0, 1, false);
    await context.eventSource.emit(context.eventTypes.CHAT_CHANGED, 'chat-two');
    assert.equal(context.setExtensionPromptCalls.at(-1)[1], '');
    context.chatId = 'chat-two';
    context.chatMetadata = { chat_id: 'chat-two' };
    context.chat = [
        { is_user: true, mes: 'question' },
        {
            is_user: false,
            mes: 'same answer',
            swipe_id: 0,
            extra: { atagia_host_message_id: 'same-local-message-id' },
        },
    ];
    await context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');

    assert.equal(calls[0].host_message_id, calls[1].host_message_id);
    assert.notEqual(calls[0].host_conversation_id, calls[1].host_conversation_id);
    assert.notEqual(calls[0].conversation_id, calls[1].conversation_id);
    module.atagiaInternals.dispose();
});

test('chat switch during assistant identity persistence cancels the stale response', async () => {
    const { context } = installSillyTavernMock();
    const saveStarted = deferred();
    const releaseSave = deferred();
    const calls = [];
    context.saveChat = async function saveChat() {
        this.saveChatCalls += 1;
        saveStarted.resolve();
        await releaseSave.promise;
    };
    globalThis.fetch = async (_url, options) => {
        calls.push(JSON.parse(options.body));
        return new Response(JSON.stringify({ message_id: 'should-not-map' }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    const chatA = [
        { is_user: true, mes: 'question A' },
        { is_user: false, mes: 'answer A', swipe_id: 0 },
    ];
    context.chat = chatA;

    const response = context.eventSource.emit(context.eventTypes.MESSAGE_RECEIVED, 1, 'normal');
    await saveStarted.promise;
    await context.eventSource.emit(context.eventTypes.CHAT_CHANGED, 'chat-B');
    context.chatId = 'chat-B';
    context.chatMetadata = { chat_id: 'chat-B' };
    context.chat = [
        { is_user: true, mes: 'question B' },
        { is_user: false, mes: 'answer B', swipe_id: 0 },
    ];
    releaseSave.resolve();
    await response;

    assert.equal(calls.length, 0);
    assert.equal(
        Object.hasOwn(chatA[1].extra.atagia_source_identity, 'atagia_message_id'),
        false,
    );
    assert.equal(Object.hasOwn(context.chat[1], 'extra'), false);
    module.atagiaInternals.dispose();
});

test('chat switch during user identity persistence cancels stale context injection', async () => {
    const { context } = installSillyTavernMock();
    const saveStarted = deferred();
    const releaseSave = deferred();
    const calls = [];
    context.saveChat = async function saveChat() {
        this.saveChatCalls += 1;
        saveStarted.resolve();
        await releaseSave.promise;
    };
    globalThis.fetch = async (_url, options) => {
        calls.push(JSON.parse(options.body));
        return new Response(JSON.stringify({ system_prompt: 'stale context' }), { status: 200 });
    };
    const module = await loadExtension();
    await module.onActivate();
    context.setExtensionPromptCalls.length = 0;
    const chatA = [{ is_user: true, mes: 'question A' }];
    context.chat = chatA;

    const interception = globalThis.atagiaMemoryInterceptor(chatA, 0, null, 'normal');
    await saveStarted.promise;
    await context.eventSource.emit(context.eventTypes.CHAT_CHANGED, 'chat-B');
    context.chatId = 'chat-B';
    context.chatMetadata = { chat_id: 'chat-B' };
    context.chat = [{ is_user: true, mes: 'question B' }];
    releaseSave.resolve();
    await interception;

    assert.equal(calls.length, 0);
    assert.equal(
        context.setExtensionPromptCalls.some(call => String(call[1]).includes('stale context')),
        false,
    );
    module.atagiaInternals.dispose();
});

test('upgrade migration purges legacy secret and private diagnostic fields', async () => {
    const oldKey = secret('legacy-browser-key');
    const alternateKey = secret('legacy-alternate-key');
    const privateMessage = secret('private-message');
    const privatePreview = secret('private-preview');
    const probe = installSillyTavernMock({
        apiKey: oldKey,
        serviceKey: alternateKey,
        baseUrl: 'http://atagia.test',
        userId: 'browser-selected-victim',
        platformId: 'browser-platform',
        lastRequest: JSON.stringify({ message_text: privateMessage }),
        lastPreview: privatePreview,
        lastRequestMessageId: 'legacy-message-id',
        lastError: privateMessage,
        lastStatus: privatePreview,
        unexpectedPersistedField: privateMessage,
    });
    const module = await loadExtension();
    await module.onActivate();

    const persisted = probe.context.extensionSettings.atagia_memory;
    for (const key of [
        'apiKey',
        'serviceKey',
        'baseUrl',
        'userId',
        'platformId',
        'lastRequest',
        'lastPreview',
        'lastRequestMessageId',
        'lastError',
        'lastStatus',
        'unexpectedPersistedField',
    ]) {
        assert.equal(Object.hasOwn(persisted, key), false, `${key} survived migration`);
    }
    assert.equal(persisted.settingsVersion, 2);
    assert.ok(probe.context.saveSettingsCalls > 0);

    const otherExtensionView = JSON.stringify(probe.context.extensionSettings);
    const renderedMarkup = module.atagiaInternals.settingsMarkup(persisted);
    const browserSurfaces = JSON.stringify({
        otherExtensionView,
        renderedMarkup,
        local: probe.localStorage.snapshot(),
        session: probe.sessionStorage.snapshot(),
    });
    for (const forbidden of [
        oldKey,
        alternateKey,
        privateMessage,
        privatePreview,
        'browser-selected-victim',
    ]) {
        assert.equal(browserSurfaces.includes(forbidden), false);
    }
    assert.equal(probe.localStorage.writes.length, 0);
    assert.equal(probe.sessionStorage.writes.length, 0);
    assert.doesNotMatch(renderedMarkup, /service api key|atagia base url|user id/i);
    module.atagiaInternals.dispose();
});

test('diagnostics are metadata-only, bounded, and cleared on reload and logout', async () => {
    const privateMessage = secret('private-debug-message');
    const privateResponse = secret('private-debug-response');
    const privatePrompt = secret('private-debug-prompt');
    const probe = installSillyTavernMock({ debug: false });
    globalThis.fetch = async () => new Response(JSON.stringify({
        system_prompt: privatePrompt,
        request_message_id: 'request-message-id',
    }), { status: 200 });
    const firstModule = await loadExtension();
    await firstModule.onActivate();
    probe.context.chat = [{ is_user: true, mes: privateMessage }];
    await globalThis.atagiaMemoryInterceptor(probe.context.chat, 0, null, 'normal');
    assert.match(probe.context.setExtensionPromptCalls.at(-1)[1], new RegExp(privatePrompt));
    firstModule.atagiaInternals.recordDiagnostic('context', 'error', {
        statusCode: 502,
        message: privateMessage,
        response: privateResponse,
        systemPrompt: privatePrompt,
    });
    assert.equal(firstModule.atagiaInternals.runtimeSnapshot().diagnostics.length, 0);

    probe.context.extensionSettings.atagia_memory.debug = true;
    for (let index = 0; index < 30; index += 1) {
        firstModule.atagiaInternals.recordDiagnostic('context', index % 2 ? 'ok' : 'error', {
            statusCode: 200 + index,
            message: privateMessage,
            response: privateResponse,
            systemPrompt: privatePrompt,
        });
    }
    const populated = firstModule.atagiaInternals.runtimeSnapshot();
    assert.equal(populated.diagnostics.length, 20);
    assert.equal(JSON.stringify(populated).includes(privateMessage), false);
    assert.equal(JSON.stringify(populated).includes(privateResponse), false);
    assert.equal(JSON.stringify(populated).includes(privatePrompt), false);

    const secondModule = await loadExtension();
    await secondModule.onActivate();
    assert.equal(firstModule.atagiaInternals.runtimeSnapshot().diagnostics.length, 0);
    assert.equal(secondModule.atagiaInternals.runtimeSnapshot().diagnostics.length, 0);
    assert.equal(probe.context.setExtensionPromptCalls.at(-1)[1], '');

    await globalThis.atagiaMemoryInterceptor(probe.context.chat, 0, null, 'normal');
    assert.match(probe.context.setExtensionPromptCalls.at(-1)[1], new RegExp(privatePrompt));
    assert.ok(secondModule.atagiaInternals.runtimeSnapshot().diagnostics.length > 0);
    const logoutListener = probe.documentListeners.get('click');
    assert.equal(typeof logoutListener, 'function');
    logoutListener({ target: { closest: selector => selector === '#logout_button' ? {} : null } });
    assert.equal(secondModule.atagiaInternals.runtimeSnapshot().diagnostics.length, 0);
    assert.equal(probe.context.setExtensionPromptCalls.at(-1)[1], '');

    await globalThis.atagiaMemoryInterceptor(probe.context.chat, 0, null, 'normal');
    assert.match(probe.context.setExtensionPromptCalls.at(-1)[1], new RegExp(privatePrompt));
    const pageHideListener = probe.windowListeners.get('pagehide');
    assert.equal(typeof pageHideListener, 'function');
    pageHideListener();
    assert.equal(secondModule.atagiaInternals.runtimeSnapshot().diagnostics.length, 0);
    assert.equal(probe.context.setExtensionPromptCalls.at(-1)[1], '');
    secondModule.atagiaInternals.dispose();
});

test('debug failure logs, browser traces, DOM markup, and bundles contain no key material', async () => {
    const oldKey = secret('compromised-copy');
    const replacementKey = secret('replacement-server-key');
    const upstreamPrivateText = secret('upstream-private');
    const probe = installSillyTavernMock({
        debug: true,
        apiKey: oldKey,
        lastRequest: upstreamPrivateText,
        lastPreview: upstreamPrivateText,
    });
    const calls = [];
    const logs = [];
    const originalWarn = console.warn;
    console.warn = message => logs.push(String(message));
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response(upstreamPrivateText, { status: 502 });
    };
    const module = await loadExtension();
    try {
        await module.onActivate();
        probe.context.chat = [{ is_user: true, mes: upstreamPrivateText }];
        await globalThis.atagiaMemoryInterceptor(probe.context.chat, 0, null, 'normal');

        const directory = path.dirname(fileURLToPath(import.meta.url));
        const bundlePaths = [
            path.join(directory, 'index.js'),
            path.join(directory, 'manifest.json'),
            path.join(directory, 'style.css'),
            path.join(directory, '..', 'server-plugin', 'index.cjs'),
            path.join(directory, '..', 'server-plugin', 'package.json'),
        ];
        const bundle = (await Promise.all(bundlePaths.map(file => fs.readFile(file, 'utf8')))).join('\n');
        const visible = JSON.stringify({
            persisted: probe.context.extensionSettings,
            local: probe.localStorage.snapshot(),
            session: probe.sessionStorage.snapshot(),
            dom: module.atagiaInternals.settingsMarkup(),
            calls,
            logs,
            runtime: module.atagiaInternals.runtimeSnapshot(),
            bundle,
        });
        for (const keyMaterial of [oldKey, replacementKey]) {
            assert.equal(visible.includes(keyMaterial), false);
        }
        assert.equal(JSON.stringify(logs).includes(upstreamPrivateText), false);
        assert.equal(JSON.stringify(probe.context.extensionSettings).includes(upstreamPrivateText), false);
        assertBrowserRequestHasNoServerAuthority(calls[0]);
    } finally {
        console.warn = originalWarn;
        module.atagiaInternals.dispose();
    }
});

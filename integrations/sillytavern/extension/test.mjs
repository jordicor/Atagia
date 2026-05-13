import assert from 'node:assert/strict';
import test from 'node:test';

function installSillyTavernMock() {
    const context = {
        extensionSettings: {
            atagia_memory: {
                enabled: true,
                baseUrl: 'http://atagia.test',
                apiKey: 'service-key',
                userId: 'usr',
                userPersonaId: 'persona',
                platformId: 'sillytavern',
                characterId: 'char',
                conversationPrefix: 'st',
                mode: 'companion',
                memoryPrivacyMode: 'trusted_private',
                debug: false,
            },
        },
        chatMetadata: { chat_id: 'chat/with slash' },
        chatId: 'chat/with slash',
        characterId: 'char',
        name1: 'persona',
        chat: [],
        saveMetadataCalls: 0,
        saveSettingsCalls: 0,
        saveMetadata() {
            this.saveMetadataCalls += 1;
        },
        saveSettingsDebounced() {
            this.saveSettingsCalls += 1;
        },
        setExtensionPromptCalls: [],
        setExtensionPrompt(...args) {
            this.setExtensionPromptCalls.push(args);
        },
        eventSource: { on() {} },
        event_types: { MESSAGE_RECEIVED: 'message_received', APP_READY: 'app_ready' },
    };
    globalThis.SillyTavern = { getContext: () => context };
    return context;
}

test('interceptor injects with setExtensionPrompt without mutating chat history', async () => {
    const context = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response(JSON.stringify({
            system_prompt: 'Remember this preference.',
            request_message_id: 'stored-user-1',
        }), { status: 200 });
    };
    await import(`./index.js?case=interceptor-${Date.now()}`);
    const chat = [{ is_user: true, mes: 'Hello', send_date: '2026-01-01T00:00:00Z' }];
    context.chat = chat;

    await globalThis.atagiaMemoryInterceptor(chat, 0, null, 'normal');

    assert.equal(chat.length, 1);
    assert.equal(context.setExtensionPromptCalls.length, 1);
    assert.match(context.setExtensionPromptCalls[0][1], /ATAGIA MEMORY CONTEXT/);
    const payload = JSON.parse(calls[0].options.body);
    assert.equal(payload.ingest_origin, 'live_turn');
    assert.equal(payload.confirmation_strategy, 'live_prompt_allowed');
    assert.equal(payload.memory_privacy_mode, 'trusted_private');
    assert.equal(typeof payload.message_id, 'string');
    assert.equal(typeof payload.source_seq, 'number');
});

test('assistant response persistence uses stable response IDs', async () => {
    const context = installSillyTavernMock();
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response('{}', { status: 200 });
    };
    const module = await import(`./index.js?case=response-${Date.now()}`);
    context.chat = [
        { is_user: true, mes: 'Hello', send_date: '2026-01-01T00:00:00Z' },
        { is_user: false, mes: 'Hi there.', send_date: '2026-01-01T00:00:01Z', swipe_id: 'a' },
    ];

    await module.atagiaInternals.recordAssistantResponse({ mes: 'Hi there.' });

    assert.equal(calls.length, 1);
    assert.match(calls[0].url, /responses$/);
    const payload = JSON.parse(calls[0].options.body);
    assert.equal(payload.text, 'Hi there.');
    assert.equal(payload.ingest_origin, 'live_turn');
    assert.equal(payload.confirmation_strategy, 'live_prompt_allowed');
    assert.equal(typeof payload.message_id, 'string');
    assert.equal(typeof payload.source_seq, 'number');
});

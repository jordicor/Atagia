import assert from 'node:assert/strict';
import test from 'node:test';

import {
    before_prompt_build,
    get_status,
    llm_output,
    session_end,
} from './index.js';

test('before_prompt_build injects Atagia context and stable IDs', async () => {
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response(JSON.stringify({
            system_prompt: 'Remember that the user likes concise answers.',
            request_message_id: 'stored-user-1',
        }), { status: 200 });
    };
    const event = await before_prompt_build(
        { config: { atagia: { baseUrl: 'http://atagia.test', apiKey: 'k' } } },
        {
            userId: 'usr',
            agentId: 'agent',
            sessionId: 'session',
            systemPrompt: 'Base system',
            messages: [{ role: 'user', content: 'Hi' }],
        },
    );

    assert.equal(calls.length, 1);
    assert.match(event.systemPrompt, /ATAGIA MEMORY CONTEXT/);
    const payload = JSON.parse(calls[0].options.body);
    assert.equal(payload.ingest_origin, 'live_turn');
    assert.equal(payload.confirmation_strategy, 'live_prompt_allowed');
    assert.equal(payload.platform_id, 'openclaw');
    assert.equal(typeof payload.message_id, 'string');
    assert.equal(typeof payload.source_seq, 'number');
    assert.equal(get_status().status, 'context_injected');
});

test('llm_output records assistant response fail-open style', async () => {
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response('{}', { status: 200 });
    };
    const event = { userId: 'usr', agentId: 'agent', sessionId: 'session', outputText: 'Done.' };
    const returned = await llm_output({}, event);

    assert.equal(returned, event);
    assert.equal(calls.length, 1);
    assert.match(calls[0].url, /responses$/);
    const payload = JSON.parse(calls[0].options.body);
    assert.equal(payload.text, 'Done.');
    assert.equal(payload.message_id.includes('assistant'), true);
    assert.equal(get_status().status, 'response_stored');
});

test('session_end backfills transcript as admin-review-only', async () => {
    const calls = [];
    globalThis.fetch = async (url, options) => {
        calls.push({ url, options });
        return new Response('{}', { status: 200 });
    };
    await session_end(
        {},
        {
            userId: 'usr',
            agentId: 'agent',
            sessionId: 'session',
            messages: [
                { role: 'user', content: 'First' },
                { role: 'assistant', content: 'Second' },
            ],
        },
    );

    assert.equal(calls.length, 2);
    assert.equal(JSON.parse(calls[0].options.body).ingest_origin, 'backfill');
    assert.equal(JSON.parse(calls[0].options.body).confirmation_strategy, 'admin_review_only');
    assert.match(calls[0].url, /messages$/);
    assert.match(calls[1].url, /responses$/);
});

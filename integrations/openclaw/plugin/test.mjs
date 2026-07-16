import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import test from 'node:test';

import plugin, {
    AtagiaOpenClawRuntime,
    SUPPORTED_OPENCLAW_VERSION,
    canonicalExternalMessageId,
    encodePathId,
} from './index.js';

const REQUIRED_CONFIG = Object.freeze({
    baseUrl: 'http://atagia.test',
    apiKey: 'service-secret',
    installationId: 'openclaw-installation',
    hostAccountId: 'openclaw-account',
    userId: 'atagia-user',
    mode: 'general_qa',
});

function responseJson(payload, status = 200) {
    return new Response(JSON.stringify(payload), { status });
}

function selectedTranscriptResponse(
    payload,
    status = 'complete',
    conversationId = FIRST_CONTEXT.sessionId,
) {
    return {
        operation_id: payload.operation_id,
        workflow_id: `workflow-${payload.operation_id}`,
        user_id: payload.user_id,
        conversation_id: conversationId,
        selection_epoch: payload.selection_epoch,
        transcript_hash: 'server-transcript-hash',
        status,
        stage: status === 'complete' ? 'complete' : status === 'remediation_required'
            ? 'remediation_required'
            : 'sources',
        selected_message_count: payload.messages?.length || 0,
        abandoned_message_count: 0,
        poll_path: `/selected-transcript/${payload.operation_id}`,
    };
}

function makeRuntime(t, { stateDir, pluginConfig = {}, responder } = {}) {
    const ownedStateDir = stateDir || fs.mkdtempSync(path.join(os.tmpdir(), 'atagia-openclaw-'));
    const calls = [];
    const warnings = [];
    const runtime = new AtagiaOpenClawRuntime({
        pluginConfig: { ...REQUIRED_CONFIG, ...pluginConfig },
        env: {},
        logger: { warn: (message) => warnings.push(message), info() {} },
        fetchImpl: async (url, options) => {
            const payload = options.body === undefined ? null : JSON.parse(options.body);
            const call = { url, options, payload };
            calls.push(call);
            if (responder) {
                return responder(call);
            }
            if (url.endsWith('/context')) {
                return responseJson({
                    request_message_id: payload.message_id,
                    system_prompt: 'Remember the user preference.',
                });
            }
            if (url.endsWith('/selected-transcript')) {
                return responseJson(selectedTranscriptResponse(payload));
            }
            return responseJson({ message_id: payload.message_id });
        },
    });
    runtime.start({ stateDir: ownedStateDir });
    t.after(() => {
        runtime.stop();
        if (!stateDir) {
            fs.rmSync(ownedStateDir, { recursive: true, force: true });
        }
    });
    return { runtime, calls, warnings, stateDir: ownedStateDir };
}

const FIRST_CONTEXT = Object.freeze({
    runId: 'run-1',
    sessionKey: 'agent:main:chat-1',
    sessionId: 'session-1',
    agentId: 'main',
});

test('native plugin registers the exact OpenClaw 2026.5.6 hooks and service', async (t) => {
    const hooks = [];
    const services = [];
    plugin.register({
        pluginConfig: { enabled: false },
        runtime: { version: '2026.5.6' },
        logger: { info() {}, warn() {} },
        on(name, handler) {
            hooks.push({ name, handler });
        },
        registerService(service) {
            services.push(service);
        },
    });

    assert.deepEqual(
        hooks.map(({ name }) => name),
        [
            'before_prompt_build',
            'agent_end',
            'before_compaction',
            'before_reset',
            'session_end',
        ],
    );
    assert.equal(services.length, 1);
    const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'atagia-openclaw-service-'));
    t.after(() => fs.rmSync(stateDir, { recursive: true, force: true }));
    await services[0].start({ stateDir });
    await services[0].stop({ stateDir });
});

test('native plugin rejects a host runtime outside the exact tested contract', () => {
    assert.throws(
        () => plugin.register({
            pluginConfig: { enabled: false },
            runtime: { version: '2026.5.7' },
            logger: { info() {}, warn() {} },
            on() {},
            registerService() {},
        }),
        /unsupported_openclaw_version/,
    );
});

test('package version metadata declares its floor and exact runtime gate', () => {
    const manifest = JSON.parse(
        fs.readFileSync(new URL('./package.json', import.meta.url), 'utf8'),
    );
    assert.equal(manifest.openclaw.install.minHostVersion, `>=${SUPPORTED_OPENCLAW_VERSION}`);
    assert.equal(manifest.openclaw.compat.pluginApi, SUPPORTED_OPENCLAW_VERSION);
    assert.equal(manifest.openclaw.build.openclawVersion, SUPPORTED_OPENCLAW_VERSION);
    assert.equal(manifest.peerDependencies.openclaw, SUPPORTED_OPENCLAW_VERSION);
});

test('before_prompt_build and agent_end use supported shapes and one complete turn', async (t) => {
    const { runtime, calls } = makeRuntime(t);
    const result = await runtime.beforePromptBuild(
        { prompt: 'Hello', messages: [] },
        FIRST_CONTEXT,
    );

    assert.deepEqual(Object.keys(result), ['prependContext']);
    assert.match(result.prependContext, /ATAGIA MEMORY CONTEXT/);
    assert.equal(calls.length, 1);
    assert.match(calls[0].url, /\/context$/);
    assert.equal(calls[0].payload.message_text, 'Hello');
    assert.equal(calls[0].payload.platform_id, 'openclaw');
    assert.equal(calls[0].payload.source_seq, 1);
    assert.match(calls[0].payload.message_id, /^extmsg_[a-f0-9]{64}$/);
    assert.equal(calls[0].payload.message_id.includes('Hello'), false);

    await runtime.agentEnd(
        {
            success: true,
            messages: [
                { role: 'user', content: 'Hello', timestamp: 1_700_000_000_000 },
                { role: 'assistant', content: [{ type: 'text', text: 'Done.' }] },
            ],
        },
        FIRST_CONTEXT,
    );

    assert.equal(calls.length, 2);
    assert.match(calls[1].url, /\/responses$/);
    assert.equal(calls[1].payload.text, 'Done.');
    assert.equal(calls[1].payload.source_seq, 2);
    assert.equal(Object.hasOwn(calls[1].payload, 'role'), false);
    assert.notEqual(calls[0].payload.message_id, calls[1].payload.message_id);

    await runtime.agentEnd(
        {
            success: true,
            messages: [
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'Done.' },
            ],
        },
        FIRST_CONTEXT,
    );
    assert.equal(calls.length, 2, 'a repeated agent_end is already durably confirmed');
    assert.equal(runtime.getStatus().status, 'response_already_stored');
});

for (const [label, acknowledgedId] of [
    ['missing', null],
    ['mismatched', 'different-message-id'],
]) {
    test(`context remains retryable when the upstream acknowledgement is ${label}`, async (t) => {
        const { runtime, calls } = makeRuntime(t, {
            responder: ({ payload }) => responseJson({
                ...(acknowledgedId === null
                    ? {}
                    : { request_message_id: acknowledgedId }),
                system_prompt: 'Unconfirmed context must not be trusted.',
            }),
        });

        assert.equal(
            await runtime.beforePromptBuild({ prompt: 'Hello', messages: [] }, FIRST_CONTEXT),
            undefined,
        );
        assert.equal(runtime.getStatus().errorCode, 'upstream_identity_mismatch');
        await runtime.beforePromptBuild({ prompt: 'Hello', messages: [] }, FIRST_CONTEXT);
        assert.equal(calls.length, 2);
        assert.equal(calls[0].payload.message_id, calls[1].payload.message_id);
    });

    test(`response remains retryable when the upstream acknowledgement is ${label}`, async (t) => {
        const { runtime, calls } = makeRuntime(t, {
            responder: ({ url, payload }) => {
                if (url.endsWith('/context')) {
                    return responseJson({ request_message_id: payload.message_id });
                }
                return responseJson(
                    acknowledgedId === null ? {} : { message_id: acknowledgedId },
                );
            },
        });
        const messages = [
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Done.' },
        ];
        await runtime.beforePromptBuild({ prompt: 'Hello', messages: [] }, FIRST_CONTEXT);
        await runtime.agentEnd({ success: true, messages }, FIRST_CONTEXT);
        assert.equal(runtime.getStatus().errorCode, 'upstream_identity_mismatch');
        await runtime.agentEnd({ success: true, messages }, FIRST_CONTEXT);
        assert.equal(calls.length, 3);
        assert.equal(calls[1].payload.message_id, calls[2].payload.message_id);
    });
}

test('later repeated text is distinct while retrying the same run is stable', async (t) => {
    const { runtime, calls } = makeRuntime(t);
    const firstMessages = [
        { role: 'user', content: 'yes' },
        { role: 'assistant', content: 'yes' },
    ];
    await runtime.beforePromptBuild({ prompt: 'yes', messages: [] }, FIRST_CONTEXT);
    await runtime.agentEnd({ success: true, messages: firstMessages }, FIRST_CONTEXT);

    const secondContext = { ...FIRST_CONTEXT, runId: 'run-2' };
    await runtime.beforePromptBuild(
        { prompt: 'yes', messages: firstMessages },
        secondContext,
    );
    await runtime.agentEnd(
        {
            success: true,
            messages: [
                ...firstMessages,
                { role: 'user', content: 'yes' },
                { role: 'assistant', content: 'yes' },
            ],
        },
        secondContext,
    );

    const payloads = calls.map(({ payload }) => payload);
    assert.deepEqual(payloads.map(({ source_seq }) => source_seq), [1, 2, 3, 4]);
    assert.equal(new Set(payloads.map(({ message_id }) => message_id)).size, 4);

    await runtime.beforePromptBuild({ prompt: 'yes', messages: [] }, FIRST_CONTEXT);
    assert.equal(calls.at(-1).payload.message_id, payloads[0].message_id);
    assert.equal(calls.at(-1).payload.source_seq, 1);
});

test('durable run mappings survive shutdown and reload', async (t) => {
    const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'atagia-openclaw-reload-'));
    t.after(() => fs.rmSync(stateDir, { recursive: true, force: true }));
    const first = makeRuntime(t, { stateDir });
    await first.runtime.beforePromptBuild({ prompt: 'same event', messages: [] }, FIRST_CONTEXT);
    const firstId = first.calls[0].payload.message_id;
    first.runtime.stop();

    const second = makeRuntime(t, { stateDir });
    await second.runtime.beforePromptBuild({ prompt: 'same event', messages: [] }, FIRST_CONTEXT);
    assert.equal(second.calls[0].payload.message_id, firstId);
    assert.equal(second.calls[0].payload.source_seq, 1);

    await second.runtime.beforePromptBuild(
        { prompt: 'same event', messages: [] },
        { ...FIRST_CONTEXT, runId: 'run-later' },
    );
    assert.notEqual(second.calls[1].payload.message_id, firstId);
    assert.equal(second.calls[1].payload.source_seq, 2);
});

test('selected transcript backfill sends the complete authoritative branch once', async (t) => {
    const { runtime, calls } = makeRuntime(t);
    await runtime.beforePromptBuild({ prompt: 'First', messages: [] }, FIRST_CONTEXT);
    await runtime.agentEnd(
        {
            success: true,
            messages: [
                { id: 'entry-user-1', role: 'user', content: 'First' },
                { id: 'entry-assistant-1', role: 'assistant', content: 'Second' },
            ],
        },
        FIRST_CONTEXT,
    );

    const transcriptPath = path.join(os.tmpdir(), `atagia-openclaw-${Date.now()}.jsonl`);
    const entries = [
        { type: 'session', version: 3, id: 'session-1', timestamp: '2026-01-01T00:00:00Z', cwd: '/tmp' },
        {
            type: 'message', id: 'entry-user-1', parentId: null, timestamp: '2026-01-01T00:00:01Z',
            message: { role: 'user', content: 'First' },
        },
        {
            type: 'message', id: 'entry-assistant-1', parentId: 'entry-user-1', timestamp: '2026-01-01T00:00:02Z',
            message: { role: 'assistant', content: 'Second' },
        },
        {
            type: 'message', id: 'entry-user-2', parentId: 'entry-assistant-1', timestamp: '2026-01-01T00:00:03Z',
            message: { role: 'user', content: 'Missing user row' },
        },
        {
            type: 'message', id: 'entry-assistant-2', parentId: 'entry-user-2', timestamp: '2026-01-01T00:00:04Z',
            message: { role: 'assistant', content: 'Missing assistant row' },
        },
    ];
    fs.writeFileSync(transcriptPath, `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`);
    t.after(() => fs.rmSync(transcriptPath, { force: true }));

    await runtime.backfill(
        { sessionFile: transcriptPath, activeLeafId: 'entry-assistant-2' },
        FIRST_CONTEXT,
        'before_compaction',
    );
    assert.equal(calls.length, 3);
    assert.match(calls[2].url, /\/selected-transcript$/);
    assert.equal(calls[2].payload.contract_version, 'atagia.selected-transcript.v1');
    assert.equal(calls[2].payload.selection_epoch, 0);
    assert.equal(calls[2].payload.mutation_kind, 'backfill');
    assert.equal(calls[2].payload.retained_cutoff_message_id, null);
    assert.deepEqual(
        calls[2].payload.messages.map((message) => [message.role, message.source_seq]),
        [['user', 3], ['assistant', 4], ['user', 5], ['assistant', 6]],
    );
    assert.deepEqual(
        calls[2].payload.messages.map((message) => message.host_message_id),
        ['entry-user-1', 'entry-assistant-1', 'entry-user-2', 'entry-assistant-2'],
    );
    assert.ok(
        calls[2].payload.messages.every(
            (message) => message.message_id.startsWith('extmsg_')
                && message.source_namespace === 'host_message',
        ),
    );

    await runtime.backfill(
        { sessionFile: transcriptPath },
        FIRST_CONTEXT,
        'session_end',
    );
    assert.equal(
        calls.length,
        3,
        'a unique terminal leaf is safely inferred without a duplicate network write',
    );
    assert.deepEqual(
        {
            imported: runtime.getStatus().imported,
            reconciled: runtime.getStatus().reconciled,
        },
        { imported: 0, reconciled: 4 },
    );
});

test('selected regeneration branch maps to the latest live run without backfill duplication', async (t) => {
    const { runtime, calls } = makeRuntime(t);
    const oldContext = { ...FIRST_CONTEXT, runId: 'run-old-generation' };
    await runtime.beforePromptBuild({ prompt: 'Question', messages: [] }, oldContext);
    await runtime.agentEnd(
        {
            success: true,
            messages: [
                { id: 'user-entry', role: 'user', content: 'Question' },
                { id: 'old-entry', role: 'assistant', content: 'Old' },
            ],
        },
        oldContext,
    );
    const newContext = { ...FIRST_CONTEXT, runId: 'run-new-generation' };
    await runtime.beforePromptBuild({ prompt: 'Question', messages: [] }, newContext);
    await runtime.agentEnd(
        {
            success: true,
            messages: [
                { id: 'user-entry', role: 'user', content: 'Question' },
                { id: 'new-entry', role: 'assistant', content: 'New' },
            ],
        },
        newContext,
    );
    assert.equal(calls.length, 4);
    assert.notEqual(calls[1].payload.message_id, calls[3].payload.message_id);

    const transcriptPath = path.join(os.tmpdir(), `atagia-openclaw-branch-${Date.now()}.jsonl`);
    const entries = [
        { type: 'session', version: 3, id: 'session-1', timestamp: '2026-01-01T00:00:00Z', cwd: '/tmp' },
        { type: 'message', id: 'user-entry', parentId: null, timestamp: '2026-01-01T00:00:01Z', message: { role: 'user', content: 'Question' } },
        { type: 'message', id: 'old-entry', parentId: 'user-entry', timestamp: '2026-01-01T00:00:02Z', message: { role: 'assistant', content: 'Old' } },
        { type: 'message', id: 'new-entry', parentId: 'user-entry', timestamp: '2026-01-01T00:00:03Z', message: { role: 'assistant', content: 'New' } },
    ];
    fs.writeFileSync(transcriptPath, `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`);
    t.after(() => fs.rmSync(transcriptPath, { force: true }));

    await runtime.backfill(
        { sessionFile: transcriptPath, activeLeafId: 'new-entry' },
        FIRST_CONTEXT,
        'session_end',
    );
    assert.equal(calls.length, 5);
    assert.match(calls[4].url, /\/selected-transcript$/);
    assert.deepEqual(
        calls[4].payload.messages.map((message) => message.host_message_id),
        ['user-entry', 'new-entry'],
    );
    assert.equal(runtime.getStatus().imported, 2);
});

test('backfill replaces an ordinal mapping when the selected host generation changes', async (t) => {
    const { runtime, calls } = makeRuntime(t);
    const transcriptPath = path.join(
        os.tmpdir(),
        `atagia-openclaw-selected-generation-${Date.now()}.jsonl`,
    );
    t.after(() => fs.rmSync(transcriptPath, { force: true }));
    const user = {
        type: 'message',
        id: 'selected-user',
        parentId: null,
        timestamp: '2026-01-01T00:00:01Z',
        message: { role: 'user', content: 'Question' },
    };
    const oldAssistant = {
        type: 'message',
        id: 'assistant-old',
        parentId: 'selected-user',
        timestamp: '2026-01-01T00:00:02Z',
        message: { role: 'assistant', content: 'Old' },
    };
    const session = {
        type: 'session',
        version: 3,
        id: 'session-1',
        timestamp: '2026-01-01T00:00:00Z',
        cwd: '/tmp',
    };
    fs.writeFileSync(
        transcriptPath,
        `${[session, user, oldAssistant].map((entry) => JSON.stringify(entry)).join('\n')}\n`,
    );
    await runtime.backfill(
        { sessionFile: transcriptPath, activeLeafId: 'assistant-old' },
        FIRST_CONTEXT,
        'session_end',
    );
    assert.equal(calls.length, 1);
    assert.deepEqual(
        calls[0].payload.messages.map((message) => message.text),
        ['Question', 'Old'],
    );

    const newAssistant = {
        type: 'message',
        id: 'assistant-new',
        parentId: 'selected-user',
        timestamp: '2026-01-01T00:00:03Z',
        message: { role: 'assistant', content: 'New' },
    };
    fs.writeFileSync(
        transcriptPath,
        `${[session, user, oldAssistant, newAssistant]
            .map((entry) => JSON.stringify(entry)).join('\n')}\n`,
    );
    await runtime.backfill(
        { sessionFile: transcriptPath, activeLeafId: 'assistant-new' },
        FIRST_CONTEXT,
        'session_end',
    );

    assert.equal(calls.length, 2);
    assert.equal(calls[1].payload.selection_epoch, 1);
    assert.equal(calls[1].payload.mutation_kind, 'regeneration');
    assert.equal(
        calls[1].payload.retained_cutoff_message_id,
        calls[0].payload.messages[0].message_id,
    );
    assert.equal(calls[1].payload.messages[1].text, 'New');
    assert.notEqual(
        calls[0].payload.messages[1].message_id,
        calls[1].payload.messages[1].message_id,
    );
    assert.deepEqual(
        { imported: runtime.getStatus().imported, reconciled: runtime.getStatus().reconciled },
        { imported: 1, reconciled: 1 },
    );
});

test('selected backfill supersedes an older live mapping not observed by a new live hook', async (t) => {
    const { runtime, calls } = makeRuntime(t);
    await runtime.beforePromptBuild({ prompt: 'Question', messages: [] }, FIRST_CONTEXT);
    await runtime.agentEnd(
        {
            success: true,
            messages: [
                { id: 'u-host', role: 'user', content: 'Question' },
                { id: 'old-host', role: 'assistant', content: 'Old' },
            ],
        },
        FIRST_CONTEXT,
    );

    await runtime.backfill(
        {
            messages: [
                { id: 'u-host', role: 'user', content: 'Question' },
                {
                    id: 'new-host',
                    generationId: 'new-generation',
                    role: 'assistant',
                    content: 'New',
                },
            ],
        },
        FIRST_CONTEXT,
        'session_end',
    );

    assert.equal(calls.length, 3);
    assert.deepEqual(
        calls.slice(0, 2).map(({ payload }) => payload.text || payload.message_text),
        ['Question', 'Old'],
    );
    assert.match(calls[2].url, /\/selected-transcript$/);
    assert.deepEqual(
        calls[2].payload.messages.map((message) => message.text),
        ['Question', 'New'],
    );
    assert.notEqual(calls[1].payload.message_id, calls[2].payload.messages[1].message_id);
    assert.deepEqual(
        { imported: runtime.getStatus().imported, reconciled: runtime.getStatus().reconciled },
        { imported: 2, reconciled: 0 },
    );
});

test('selected rebuild polls, performs one remediation retry, and advances epochs durably', async (t) => {
    let submitted = null;
    let statusReads = 0;
    const { runtime, calls, stateDir } = makeRuntime(t, {
        pluginConfig: {
            selectedTranscriptWaitMs: 500,
            selectedTranscriptPollIntervalMs: 25,
        },
        responder: ({ url, options, payload }) => {
            if (url.endsWith('/selected-transcript')) {
                submitted = payload;
                return responseJson(selectedTranscriptResponse(payload, 'rebuilding'), 202);
            }
            if (url.endsWith('/retry')) {
                assert.equal(payload.user_id, REQUIRED_CONFIG.userId);
                return responseJson(selectedTranscriptResponse(submitted, 'rebuilding'), 202);
            }
            if (options.method === 'GET' && url.includes('/selected-transcript/')) {
                statusReads += 1;
                return responseJson(
                    selectedTranscriptResponse(
                        submitted,
                        statusReads === 1 ? 'remediation_required' : 'complete',
                    ),
                );
            }
            throw new Error(`unexpected request ${options.method} ${url}`);
        },
    });
    const selected = [
        { id: 'user-v1', role: 'user', content: 'Question' },
        { id: 'assistant-v1', role: 'assistant', content: 'Answer one' },
    ];
    await runtime.backfill({ messages: selected }, FIRST_CONTEXT, 'session_end');

    assert.deepEqual(calls.map(({ options }) => options.method), ['POST', 'GET', 'POST', 'GET']);
    assert.match(calls[0].url, /\/selected-transcript$/);
    assert.match(calls[2].url, /\/retry$/);
    assert.equal(calls[0].payload.selection_epoch, 0);
    assert.equal(runtime.getStatus().status, 'selected_transcript_complete');

    const persisted = fs.readFileSync(
        path.join(stateDir, 'plugins', 'atagia-memory', 'identity.json'),
        'utf8',
    );
    assert.equal(persisted.includes('Question'), false);
    assert.equal(persisted.includes('Answer one'), false);

    await runtime.backfill(
        {
            messages: [
                selected[0],
                { id: 'assistant-v2', role: 'assistant', content: 'Answer two' },
            ],
        },
        FIRST_CONTEXT,
        'session_end',
    );
    const secondSelection = calls.findLast(
        ({ url, payload }) => url.endsWith('/selected-transcript')
            && payload.selection_epoch === 1,
    );
    assert.ok(secondSelection);
    assert.notEqual(secondSelection.payload.operation_id, calls[0].payload.operation_id);
    assert.equal(secondSelection.payload.mutation_kind, 'regeneration');
});

test('a dropped remediation retry remains recoverable after restart', async (t) => {
    const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'atagia-openclaw-remediation-'));
    t.after(() => fs.rmSync(stateDir, { recursive: true, force: true }));
    const selected = [
        { id: 'remediation-user', role: 'user', content: 'Repair request' },
        { id: 'remediation-assistant', role: 'assistant', content: 'Repair answer' },
    ];
    let submitted = null;
    const first = makeRuntime(t, {
        stateDir,
        pluginConfig: {
            selectedTranscriptWaitMs: 500,
            selectedTranscriptPollIntervalMs: 25,
        },
        responder: ({ url, payload }) => {
            if (url.endsWith('/selected-transcript')) {
                submitted = payload;
                return responseJson(selectedTranscriptResponse(payload, 'remediation_required'));
            }
            if (url.endsWith('/retry')) {
                throw new Error('connection dropped before retry acknowledgement');
            }
            throw new Error(`unexpected request ${url}`);
        },
    });
    await first.runtime.backfill({ messages: selected }, FIRST_CONTEXT, 'session_end');
    assert.equal(first.runtime.getStatus().errorCode, 'upstream_unavailable');
    assert.deepEqual(first.calls.map(({ options }) => options.method), ['POST', 'POST']);
    const originalOperationId = submitted.operation_id;
    const persistedAfterDrop = JSON.parse(fs.readFileSync(
        path.join(stateDir, 'plugins', 'atagia-memory', 'identity.json'),
        'utf8',
    ));
    const [scopeAfterDrop] = Object.values(persistedAfterDrop.scopes);
    assert.equal(scopeAfterDrop.selection.pending.remediationRetries, 0);
    first.runtime.stop();

    let statusReads = 0;
    const second = makeRuntime(t, {
        stateDir,
        pluginConfig: {
            selectedTranscriptWaitMs: 500,
            selectedTranscriptPollIntervalMs: 25,
        },
        responder: ({ url, options }) => {
            if (options.method === 'GET') {
                statusReads += 1;
                return responseJson(selectedTranscriptResponse(
                    submitted,
                    statusReads === 1 ? 'remediation_required' : 'complete',
                ));
            }
            if (url.endsWith('/retry')) {
                return responseJson(selectedTranscriptResponse(submitted, 'rebuilding'), 202);
            }
            throw new Error(`unexpected request ${options.method} ${url}`);
        },
    });
    await second.runtime.backfill({ messages: selected }, FIRST_CONTEXT, 'session_end');

    assert.deepEqual(
        second.calls.map(({ options }) => options.method),
        ['GET', 'POST', 'GET'],
    );
    assert.match(second.calls[1].url, /\/retry$/);
    assert.ok(second.calls.every(({ url }) => url.includes(originalOperationId)));
    assert.equal(second.runtime.getStatus().status, 'selected_transcript_complete');
});

test('startup resumes an accepted old-session reset workflow without another old hook', async (t) => {
    const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'atagia-openclaw-reset-recovery-'));
    t.after(() => fs.rmSync(stateDir, { recursive: true, force: true }));
    const selected = [
        { id: 'reset-user', role: 'user', content: 'Before reset' },
        { id: 'reset-assistant', role: 'assistant', content: 'Old session answer' },
    ];
    let submitted = null;
    const first = makeRuntime(t, {
        stateDir,
        pluginConfig: {
            selectedTranscriptWaitMs: 100,
            selectedTranscriptPollIntervalMs: 25,
        },
        responder: ({ url, options, payload }) => {
            if (url.endsWith('/selected-transcript')) {
                submitted = payload;
                return responseJson(selectedTranscriptResponse(payload, 'rebuilding'), 202);
            }
            if (options.method === 'GET') {
                return responseJson(selectedTranscriptResponse(submitted, 'rebuilding'), 202);
            }
            throw new Error(`unexpected request ${options.method} ${url}`);
        },
    });
    await first.runtime.backfill({ messages: selected }, FIRST_CONTEXT, 'before_reset');
    assert.equal(first.runtime.getStatus().status, 'selected_transcript_rebuilding');
    const persistedPending = JSON.parse(fs.readFileSync(
        path.join(stateDir, 'plugins', 'atagia-memory', 'identity.json'),
        'utf8',
    ));
    const [oldScope] = Object.values(persistedPending.scopes);
    assert.deepEqual(oldScope.identity, {
        installationId: REQUIRED_CONFIG.installationId,
        hostAccountId: REQUIRED_CONFIG.hostAccountId,
        userId: REQUIRED_CONFIG.userId,
        hostConversationId: FIRST_CONTEXT.sessionId,
    });
    assert.equal(oldScope.selection.pending.serverAccepted, true);
    assert.equal(Object.hasOwn(oldScope.selection.pending, 'requestMessages'), false);
    first.runtime.stop();

    let oldStatusReads = 0;
    const second = makeRuntime(t, {
        stateDir,
        pluginConfig: {
            selectedTranscriptWaitMs: 500,
            selectedTranscriptPollIntervalMs: 25,
        },
        responder: ({ url, options, payload }) => {
            if (url.endsWith('/context')) {
                return responseJson({
                    request_message_id: payload.message_id,
                    system_prompt: '',
                });
            }
            if (options.method === 'GET') {
                oldStatusReads += 1;
                return responseJson(selectedTranscriptResponse(
                    submitted,
                    oldStatusReads === 1 ? 'remediation_required' : 'complete',
                ));
            }
            if (url.endsWith('/retry')) {
                return responseJson(selectedTranscriptResponse(submitted, 'rebuilding'), 202);
            }
            throw new Error(`unexpected request ${options.method} ${url}`);
        },
    });
    const recovery = await second.runtime.resumePendingSelections();
    assert.deepEqual(recovery, { attempted: 1, completed: 1 });
    assert.deepEqual(
        second.calls.map(({ options }) => options.method),
        ['GET', 'POST', 'GET'],
    );
    assert.ok(second.calls.every(({ url }) => url.includes(FIRST_CONTEXT.sessionId)));
    assert.equal(
        second.runtime.getStatus().status,
        'selected_transcript_recovery_complete',
    );

    const newContext = {
        ...FIRST_CONTEXT,
        runId: 'run-after-reset',
        sessionId: 'session-after-reset',
    };
    await second.runtime.beforePromptBuild(
        { prompt: 'New session prompt', messages: [] },
        newContext,
    );
    assert.match(second.calls.at(-1).url, /\/session-after-reset\/context$/);
});

test('startup resubmits an unacknowledged selection outbox with the same operation id', async (t) => {
    const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'atagia-openclaw-selection-retry-'));
    t.after(() => fs.rmSync(stateDir, { recursive: true, force: true }));
    const selected = [
        { id: 'retry-user', role: 'user', content: 'Stable request' },
        { id: 'retry-assistant', role: 'assistant', content: 'Stable answer' },
    ];
    const first = makeRuntime(t, {
        stateDir,
        responder: ({ url }) => {
            if (url.endsWith('/selected-transcript')) {
                throw new Error('connection dropped before acknowledgement');
            }
            throw new Error(`unexpected request ${url}`);
        },
    });
    await first.runtime.backfill({ messages: selected }, FIRST_CONTEXT, 'session_end');
    assert.equal(first.runtime.getStatus().errorCode, 'upstream_unavailable');
    const original = first.calls[0].payload;
    const persistedOutbox = fs.readFileSync(
        path.join(stateDir, 'plugins', 'atagia-memory', 'identity.json'),
        'utf8',
    );
    assert.match(persistedOutbox, /Stable request/);
    first.runtime.stop();

    const second = makeRuntime(t, {
        stateDir,
        responder: ({ url, options, payload }) => {
            if (options.method === 'GET') {
                return responseJson({ detail: 'not found' }, 404);
            }
            if (url.endsWith('/selected-transcript')) {
                return responseJson(selectedTranscriptResponse(payload));
            }
            throw new Error(`unexpected request ${options.method} ${url}`);
        },
    });
    const recovery = await second.runtime.resumePendingSelections();
    assert.deepEqual(recovery, { attempted: 1, completed: 1 });
    assert.deepEqual(second.calls.map(({ options }) => options.method), ['GET', 'POST']);
    assert.equal(second.calls[1].payload.operation_id, original.operation_id);
    assert.equal(second.calls[1].payload.selection_epoch, original.selection_epoch);
    assert.deepEqual(second.calls[1].payload.messages, original.messages);
    const persistedComplete = fs.readFileSync(
        path.join(stateDir, 'plugins', 'atagia-memory', 'identity.json'),
        'utf8',
    );
    assert.equal(persistedComplete.includes('Stable request'), false);
});

test('an incomplete selected turn is deferred without destructive reconciliation', async (t) => {
    const { runtime, calls } = makeRuntime(t);
    await runtime.backfill(
        { messages: [{ id: 'pending-user', role: 'user', content: 'Still running' }] },
        FIRST_CONTEXT,
        'before_compaction',
    );
    assert.equal(calls.length, 0);
    assert.equal(runtime.getStatus().status, 'failed_open');
    assert.equal(runtime.getStatus().errorCode, 'selected_transcript_incomplete');
});

test('session rollover isolates two transcripts that reuse the same session key', async (t) => {
    const { runtime, calls } = makeRuntime(t, {
        responder: ({ url, payload }) => {
            const match = url.match(/\/v1\/conversations\/([^/]+)\/selected-transcript$/);
            assert.ok(match);
            return responseJson(
                selectedTranscriptResponse(payload, 'complete', match[1]),
                202,
            );
        },
    });
    const routingSessionKey = 'agent:main:reused-route';
    const records = [
        { id: 'same-user-entry', role: 'user', content: 'Question' },
        { id: 'same-assistant-entry', role: 'assistant', content: 'Answer' },
    ];
    await runtime.backfill(
        { sessionId: 'session-before-reset', messages: records },
        { sessionKey: routingSessionKey, sessionId: 'session-before-reset' },
        'session_end',
    );
    await runtime.backfill(
        { sessionId: 'session-after-reset', messages: records },
        { sessionKey: routingSessionKey, sessionId: 'session-after-reset' },
        'session_end',
    );

    assert.deepEqual(
        calls.map(({ url }) => url),
        [
            'http://atagia.test/v1/conversations/session-before-reset/selected-transcript',
            'http://atagia.test/v1/conversations/session-after-reset/selected-transcript',
        ],
    );
    assert.deepEqual(calls.map(({ payload }) => payload.selection_epoch), [0, 0]);
    assert.notEqual(calls[0].payload.operation_id, calls[1].payload.operation_id);
    assert.notEqual(
        calls[0].payload.messages[0].message_id,
        calls[1].payload.messages[0].message_id,
    );
});

test('selected reconciliation defers when only a reusable session key is known', async (t) => {
    const { runtime, calls } = makeRuntime(t);
    await runtime.backfill(
        {
            messages: [
                { id: 'user-entry', role: 'user', content: 'Question' },
                { id: 'assistant-entry', role: 'assistant', content: 'Answer' },
            ],
        },
        { sessionKey: 'agent:main:routing-only' },
        'session_end',
    );
    assert.equal(calls.length, 0);
    assert.equal(runtime.getStatus().status, 'selected_transcript_deferred');
    assert.equal(runtime.getStatus().errorCode, 'host_session_id_required');
});

test('before_compaction accepts the first JSONL session record as canonical id', async (t) => {
    const transcriptPath = path.join(os.tmpdir(), `atagia-openclaw-header-${Date.now()}.jsonl`);
    t.after(() => fs.rmSync(transcriptPath, { force: true }));
    const entries = [
        { type: 'session', id: 'session-from-header', version: 3 },
        {
            type: 'message', id: 'header-user', parentId: null,
            message: { role: 'user', content: 'Question' },
        },
        {
            type: 'message', id: 'header-assistant', parentId: 'header-user',
            message: { role: 'assistant', content: 'Answer' },
        },
    ];
    fs.writeFileSync(
        transcriptPath,
        `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`,
    );
    const { runtime, calls } = makeRuntime(t, {
        responder: ({ payload }) => responseJson(
            selectedTranscriptResponse(payload, 'complete', 'session-from-header'),
            202,
        ),
    });
    await runtime.backfill(
        { sessionFile: transcriptPath, activeLeafId: 'header-assistant' },
        { sessionKey: 'agent:main:routing-only' },
        'before_compaction',
    );
    assert.equal(calls.length, 1);
    assert.match(calls[0].url, /\/conversations\/session-from-header\/selected-transcript$/);
});

test('before_reset active messages select an older JSONL branch instead of append order', async (t) => {
    const transcriptPath = path.join(os.tmpdir(), `atagia-openclaw-navigation-${Date.now()}.jsonl`);
    t.after(() => fs.rmSync(transcriptPath, { force: true }));
    const entries = [
        { type: 'session', id: 'navigation-session', version: 3 },
        {
            type: 'message', id: 'navigation-user', parentId: null,
            message: { role: 'user', content: 'Question' },
        },
        {
            type: 'message', id: 'active-assistant', parentId: 'navigation-user',
            message: { role: 'assistant', content: 'Active answer' },
        },
        {
            type: 'message', id: 'abandoned-assistant', parentId: 'navigation-user',
            message: { role: 'assistant', content: 'Abandoned later append' },
        },
    ];
    fs.writeFileSync(
        transcriptPath,
        `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`,
    );
    const { runtime, calls } = makeRuntime(t, {
        responder: ({ payload }) => responseJson(
            selectedTranscriptResponse(payload, 'complete', 'navigation-session'),
            202,
        ),
    });
    await runtime.backfill(
        {
            sessionFile: transcriptPath,
            messages: [
                { role: 'user', content: 'Question' },
                { role: 'assistant', content: 'Active answer' },
            ],
        },
        { sessionKey: 'agent:main:reused-route', sessionId: 'navigation-session' },
        'before_reset',
    );
    assert.deepEqual(
        calls[0].payload.messages.map((message) => message.host_message_id),
        ['navigation-user', 'active-assistant'],
    );

    await runtime.backfill(
        { sessionId: 'navigation-session', sessionFile: transcriptPath },
        { sessionKey: 'agent:main:reused-route', sessionId: 'navigation-session' },
        'session_end',
    );
    assert.equal(calls.length, 1);
    assert.equal(runtime.getStatus().status, 'selected_transcript_deferred');
    assert.equal(runtime.getStatus().errorCode, 'active_transcript_leaf_required');
});

test('deep active-branch reconstruction remains linear within the hook budget', async (t) => {
    const transcriptPath = path.join(os.tmpdir(), `atagia-openclaw-deep-${Date.now()}.jsonl`);
    t.after(() => fs.rmSync(transcriptPath, { force: true }));
    const messageCount = 10_001;
    const entries = [{ type: 'session', id: 'deep-session', version: 3 }];
    const messages = [];
    let parentId = null;
    for (let index = 0; index < messageCount; index += 1) {
        const id = `deep-entry-${index}`;
        const role = index % 2 === 0 ? 'user' : 'assistant';
        const content = `${role}-${index}`;
        entries.push({
            type: 'message',
            id,
            parentId,
            message: { role, content },
        });
        messages.push({ role, content });
        parentId = id;
    }
    fs.writeFileSync(
        transcriptPath,
        `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`,
    );
    assert.ok(fs.statSync(transcriptPath).size < 64 * 1024 * 1024);

    const { runtime, calls } = makeRuntime(t);
    const startedAt = performance.now();
    await runtime.backfill(
        { sessionFile: transcriptPath, messages },
        { ...FIRST_CONTEXT, sessionId: 'deep-session' },
        'before_compaction',
    );
    const elapsedMs = performance.now() - startedAt;

    assert.ok(elapsedMs < 3_000, `deep reconstruction took ${elapsedMs.toFixed(0)} ms`);
    assert.equal(calls.length, 0);
    assert.equal(runtime.getStatus().errorCode, 'selected_transcript_incomplete');
});

test('canonical identity has all global namespace dimensions and no text input', () => {
    const base = {
        installationId: 'installation-1',
        hostAccountId: 'account-1',
        mappedUserId: 'user-1',
        hostConversationId: 'conversation-1',
        sourceNamespace: 'host_message',
        hostMessageId: 'message-1',
        role: 'assistant',
        generationId: 'generation-1',
    };
    const original = canonicalExternalMessageId(base);
    assert.equal(original, canonicalExternalMessageId({ ...base }));
    for (const [key, value] of [
        ['installationId', 'installation-2'],
        ['hostAccountId', 'account-2'],
        ['mappedUserId', 'user-2'],
        ['hostConversationId', 'conversation-2'],
        ['sourceNamespace', 'live_event'],
        ['hostMessageId', 'message-2'],
        ['role', 'user'],
        ['generationId', 'generation-2'],
        ['integrationKind', 'hermes'],
    ]) {
        assert.notEqual(canonicalExternalMessageId({ ...base, [key]: value }), original);
    }
});

test('transport IDs match Atagia reversible path rules', () => {
    assert.equal(encodePathId('safe:id-1'), 'safe:id-1');
    assert.match(encodePathId('chat/日本語'), /^__atagia_b64_[A-Za-z0-9_-]+$/);
    assert.match(encodePathId('__atagia_b64_reserved'), /^__atagia_b64_[A-Za-z0-9_-]+$/);
});

test('fail-open state and logs retain metadata only', async (t) => {
    const secretBody = 'private upstream body and prompt text';
    const { runtime, warnings } = makeRuntime(t, {
        responder: () => new Response(secretBody, { status: 503 }),
    });
    const result = await runtime.beforePromptBuild(
        { prompt: 'private prompt text', messages: [] },
        FIRST_CONTEXT,
    );
    assert.equal(result, undefined);
    const serialized = JSON.stringify(runtime.getStatus());
    assert.equal(serialized.includes('private'), false);
    assert.equal(serialized.includes(secretBody), false);
    assert.equal(runtime.getStatus().errorCode, 'upstream_http_503');
    assert.equal(warnings.join('\n').includes('private'), false);
});

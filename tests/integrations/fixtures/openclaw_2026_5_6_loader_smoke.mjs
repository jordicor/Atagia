import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [hostRootArg, pluginRootArg] = process.argv.slice(2);
assert.ok(hostRootArg, 'OpenClaw checkout path is required');
assert.ok(pluginRootArg, 'Atagia plugin path is required');

const hostRoot = path.resolve(hostRootArg);
const pluginRoot = path.resolve(pluginRootArg);
const stateDir = fs.mkdtempSync(path.join(os.tmpdir(), 'atagia-openclaw-loader-'));
process.env.OPENCLAW_STATE_DIR = stateDir;
process.env.OPENCLAW_DISABLE_BUNDLED_PLUGINS = '1';

const requests = [];
const server = http.createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) {
        chunks.push(chunk);
    }
    const payload = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    requests.push({ url: request.url, payload });
    let result;
    if (request.url.endsWith('/context')) {
        result = {
            request_message_id: payload.message_id,
            system_prompt: 'Context returned through the official OpenClaw loader.',
        };
    } else if (request.url.endsWith('/selected-transcript')) {
        result = {
            operation_id: payload.operation_id,
            workflow_id: `workflow-${payload.operation_id}`,
            user_id: payload.user_id,
            conversation_id: 'official-loader-session',
            selection_epoch: payload.selection_epoch,
            transcript_hash: 'loader-transcript-hash',
            status: 'complete',
            stage: 'complete',
            selected_message_count: payload.messages.length,
            abandoned_message_count: 0,
            poll_path: `/selected-transcript/${payload.operation_id}`,
        };
    } else {
        result = { message_id: payload.message_id };
    }
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify(result));
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const address = server.address();
assert.ok(address && typeof address === 'object');

const config = {
    plugins: {
        allow: ['atagia-memory'],
        load: { paths: [pluginRoot] },
        entries: {
            'atagia-memory': {
                enabled: true,
                hooks: { allowConversationAccess: true },
                config: {
                    baseUrl: `http://127.0.0.1:${address.port}`,
                    apiKey: 'loader-service-key',
                    installationId: 'loader-installation',
                    hostAccountId: 'loader-account',
                    userId: 'loader-atagia-user',
                },
            },
        },
    },
};

const loaderUrl = pathToFileURL(path.join(hostRoot, 'dist', 'plugins', 'loader.js')).href;
const {
    clearActivatedPluginRuntimeState,
    clearPluginLoaderCache,
    loadOpenClawPlugins,
} = await import(loaderUrl);

function loadRegistry() {
    clearPluginLoaderCache();
    clearActivatedPluginRuntimeState();
    return loadOpenClawPlugins({
        config,
        onlyPluginIds: ['atagia-memory'],
        cache: false,
    });
}

async function startServices(registry) {
    const running = [];
    for (const registration of registry.services) {
        await registration.service.start({
            config,
            stateDir,
            logger: { info() {}, warn() {}, error() {}, debug() {} },
        });
        running.push(registration.service);
    }
    return async () => {
        for (const service of running.toReversed()) {
            await service.stop?.({
                config,
                stateDir,
                logger: { info() {}, warn() {}, error() {}, debug() {} },
            });
        }
    };
}

try {
    const firstRegistry = loadRegistry();
    const plugin = firstRegistry.plugins.find((entry) => entry.id === 'atagia-memory');
    assert.equal(plugin?.status, 'loaded');
    const hookNames = firstRegistry.typedHooks
        .filter((entry) => entry.pluginId === 'atagia-memory')
        .map((entry) => entry.hookName);
    assert.deepEqual(
        hookNames,
        [
            'before_prompt_build',
            'agent_end',
            'before_compaction',
            'before_reset',
            'session_end',
        ],
    );
    assert.equal(firstRegistry.services.length, 1);
    const stopFirst = await startServices(firstRegistry);
    const beforePrompt = firstRegistry.typedHooks.find(
        (entry) => entry.pluginId === 'atagia-memory' && entry.hookName === 'before_prompt_build',
    );
    const agentEnd = firstRegistry.typedHooks.find(
        (entry) => entry.pluginId === 'atagia-memory' && entry.hookName === 'agent_end',
    );
    const sessionEnd = firstRegistry.typedHooks.find(
        (entry) => entry.pluginId === 'atagia-memory' && entry.hookName === 'session_end',
    );
    const beforeCompaction = firstRegistry.typedHooks.find(
        (entry) => entry.pluginId === 'atagia-memory' && entry.hookName === 'before_compaction',
    );
    const beforeReset = firstRegistry.typedHooks.find(
        (entry) => entry.pluginId === 'atagia-memory' && entry.hookName === 'before_reset',
    );
    assert.ok(beforePrompt && agentEnd && beforeCompaction && beforeReset && sessionEnd);
    const ctx = {
        runId: 'official-loader-run',
        sessionKey: 'agent:main:official-loader',
        sessionId: 'official-loader-session',
        agentId: 'main',
    };
    const promptResult = await beforePrompt.handler(
        { prompt: 'Official loader turn', messages: [] },
        ctx,
    );
    assert.match(promptResult.prependContext, /official OpenClaw loader/);
    await agentEnd.handler(
        {
            success: true,
            messages: [
                { role: 'user', content: 'Official loader turn' },
                { role: 'assistant', content: 'Official loader response' },
            ],
        },
        ctx,
    );
    assert.equal(requests.length, 2);
    assert.match(requests[0].url, /\/context$/);
    assert.match(requests[1].url, /\/responses$/);
    assert.equal(requests[0].payload.source_seq, 1);
    assert.equal(requests[1].payload.source_seq, 2);
    const originalId = requests[0].payload.message_id;
    const transcriptPath = path.join(stateDir, 'official-loader-session.jsonl');
    fs.writeFileSync(
        transcriptPath,
        `${[
            {
                type: 'session',
                version: 3,
                id: 'official-loader-session',
            },
            {
                type: 'message',
                id: 'official-loader-user',
                parentId: null,
                message: { role: 'user', content: 'Official loader turn' },
            },
            {
                type: 'message',
                id: 'official-loader-assistant',
                parentId: 'official-loader-user',
                message: { role: 'assistant', content: 'Official loader response' },
            },
        ].map((entry) => JSON.stringify(entry)).join('\n')}\n`,
    );
    await beforeReset.handler(
        {
            messages: [
                { role: 'user', content: 'Official loader turn' },
                { role: 'assistant', content: 'Official loader response' },
            ],
            reason: 'reset',
            sessionFile: transcriptPath,
        },
        ctx,
    );
    await beforeCompaction.handler(
        {
            messageCount: 2,
            messages: [
                { role: 'user', content: 'Official loader turn' },
                { role: 'assistant', content: 'Official loader response' },
            ],
            sessionFile: transcriptPath,
        },
        ctx,
    );
    await sessionEnd.handler(
        {
            sessionId: 'official-loader-session',
            sessionKey: ctx.sessionKey,
            messageCount: 2,
            reason: 'deleted',
            sessionFile: transcriptPath,
        },
        ctx,
    );
    assert.equal(requests.length, 3);
    assert.equal(
        requests[2].url,
        '/v1/conversations/official-loader-session/selected-transcript',
    );
    assert.deepEqual(
        requests[2].payload.messages.map((message) => message.host_message_id),
        ['official-loader-user', 'official-loader-assistant'],
    );
    assert.equal(
        requests.length,
        3,
        'linear session_end infers the unique leaf without a duplicate write',
    );
    await stopFirst();

    const reloadedRegistry = loadRegistry();
    const stopReloaded = await startServices(reloadedRegistry);
    const reloadedBeforePrompt = reloadedRegistry.typedHooks.find(
        (entry) => entry.pluginId === 'atagia-memory' && entry.hookName === 'before_prompt_build',
    );
    assert.ok(reloadedBeforePrompt);
    await reloadedBeforePrompt.handler(
        { prompt: 'Official loader turn', messages: [] },
        ctx,
    );
    assert.equal(requests.at(-1).payload.message_id, originalId);
    await stopReloaded();

    process.stdout.write(`${JSON.stringify({
        status: plugin.status,
        hookNames,
        lifecycleRequests: requests.length,
        reloadStable: true,
    })}\n`);
} finally {
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(stateDir, { recursive: true, force: true });
}

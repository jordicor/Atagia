'use strict';

const http = require('node:http');
const path = require('node:path');

const pluginPath = path.resolve(process.argv[2]);
const plugin = require(pluginPath);
const routePrefix = `/api/plugins/${plugin.info.id}`;
const handle = process.env.ATAGIA_TEST_SILLYTAVERN_HANDLE || 'alice';
const csrfToken = process.env.ATAGIA_TEST_CSRF_TOKEN || 'test-csrf-token';

function routerFixture() {
    const middleware = [];
    const routes = new Map();
    return {
        routes,
        use(handler) {
            middleware.push(handler);
        },
        post(routePath, ...handlers) {
            routes.set(`POST ${routePath}`, [...middleware, ...handlers]);
        },
    };
}

async function readJson(request) {
    const chunks = [];
    let size = 0;
    for await (const chunk of request) {
        size += chunk.length;
        if (size > 2 * 1024 * 1024) {
            throw new Error('request_too_large');
        }
        chunks.push(chunk);
    }
    if (chunks.length === 0) {
        return {};
    }
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function responseFixture(response) {
    return {
        status(statusCode) {
            response.statusCode = statusCode;
            return this;
        },
        json(payload) {
            if (!response.headersSent) {
                response.setHeader('Content-Type', 'application/json');
            }
            response.end(JSON.stringify(payload));
            return this;
        },
        sendStatus(statusCode) {
            response.statusCode = statusCode;
            response.end();
            return this;
        },
    };
}

const router = routerFixture();

async function start() {
    await plugin.init(router);
    const server = http.createServer(async (request, response) => {
        try {
            const url = new URL(request.url, 'http://127.0.0.1');
            if (!url.pathname.startsWith(routePrefix)) {
                response.statusCode = 404;
                response.end();
                return;
            }
            const routePath = url.pathname.slice(routePrefix.length) || '/';
            const handlers = router.routes.get(`${request.method} ${routePath}`);
            if (!handlers) {
                response.statusCode = 404;
                response.end();
                return;
            }
            request.body = await readJson(request);
            request.user = { profile: { handle } };
            request.session = { handle, csrfToken };
            request.get = name => request.headers[name.toLowerCase()];
            const wrappedResponse = responseFixture(response);
            let index = 0;
            async function next() {
                const handler = handlers[index++];
                if (!handler) {
                    return;
                }
                return handler(request, wrappedResponse, next);
            }
            await next();
            if (!response.writableEnded) {
                response.statusCode = 204;
                response.end();
            }
        } catch {
            if (!response.writableEnded) {
                response.statusCode = 400;
                response.setHeader('Content-Type', 'application/json');
                response.end(JSON.stringify({ error: 'invalid_request' }));
            }
        }
    });

    server.listen(0, '127.0.0.1', () => {
        const address = server.address();
        process.stdout.write(`ATAGIA_TEST_SERVER_PORT=${address.port}\n`);
    });

    let stopping = false;
    async function stop() {
        if (stopping) {
            return;
        }
        stopping = true;
        await plugin.exit();
        server.close(() => process.exit(0));
    }
    process.on('SIGTERM', stop);
    process.on('SIGINT', stop);
    process.stdin.on('data', data => {
        if (String(data).trim() === 'stop') {
            stop();
        }
    });
}

start().catch(() => process.exit(1));

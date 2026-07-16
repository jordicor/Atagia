# Atagia Memory Server Plugin

Compatibility target: SillyTavern `1.18.0` at commit
`51ad27fb86d39a3daca3adaa970375c9670c12df`.

This is the trusted same-origin boundary for the Atagia Memory browser
extension. It holds one Atagia service key in server memory and derives the
Atagia user from the authenticated SillyTavern user. It never returns the key or
user mapping to the browser.

The code and local boundary tests are complete. This is a reference
integration: it is not yet production-supported until a real assistant-response
turn is verified against the target SillyTavern install. Installing this plugin
does not itself provision or replace any credential.

## Install

Copy this directory to:

```text
<SillyTavern>/plugins/atagia-memory
```

Pin SillyTavern to the compatibility target and set:

```yaml
enableServerPlugins: true
enableServerPluginsAutoUpdate: false
disableCsrfProtection: false
```

Provide server-only environment values:

```bash
ATAGIA_BASE_URL=https://atagia.internal.example
ATAGIA_SERVICE_API_KEY=<replacement-key-from-secret-manager>
ATAGIA_SILLYTAVERN_INSTALLATION_ID=<stable-unique-deployment-id>
ATAGIA_SILLYTAVERN_USER_MAP='{"alice":"atagia-user-alice","bob":"atagia-user-bob"}'
ATAGIA_SILLYTAVERN_TIMEOUT_MS=20000
```

`ATAGIA_SERVICE_API_KEY`, `ATAGIA_SILLYTAVERN_INSTALLATION_ID`, and
`ATAGIA_SILLYTAVERN_USER_MAP` are mandatory. The installation ID is non-secret
but must remain stable and unique to this host deployment.
`ATAGIA_BASE_URL` defaults to `http://127.0.0.1:8100` and must be an HTTP(S) URL
without embedded credentials, query, or fragment. The timeout is optional and
must be from 100 through 120000 milliseconds.

Keep the environment and process logs server-side. The startup log contains
only a short SHA-256 credential fingerprint and mapping count. It must not be
used as a substitute for secret-manager inventory.

Restart SillyTavern after configuration. The official server-plugin loader
mounts these routes beneath `/api/plugins/atagia-memory`:

- `POST /health`
- `POST /context`
- `POST /response`

## Boundary Rules

Every request must have:

- an authenticated `request.user.profile.handle`;
- a matching SillyTavern session;
- a valid `X-CSRF-Token` matching the session token; and
- same-origin fetch metadata when the browser supplies `Sec-Fetch-Site`.

The authenticated handle must have an exact entry in
`ATAGIA_SILLYTAVERN_USER_MAP`. The plugin rejects JSON fields that attempt to
provide authorization, a service key, Atagia user identity, or platform
identity, and it never forwards corresponding browser headers. It adds the
mapped user, fixed `sillytavern` platform, service credential, ingest origin,
and confirmation strategy only on the server-side Atagia request.

Responses are allowlisted: health returns status only; context returns the
system prompt plus canonical request mapping; response ingest returns only the
canonical message ID and optional source sequence needed to persist the host
mapping. Upstream error bodies and credentials are never
forwarded or logged. Requests use a bounded timeout, bounded fields and response
size, and active upstream calls are aborted when the plugin exits.

The browser supplies only host conversation/message/generation identity and an
optional chronological ordinal. The server combines it with the configured
installation, authenticated handle, mapped Atagia user, fixed integration, and
role to create the canonical Atagia message ID. Browser-supplied final message
IDs are rejected, and text is never hashed into identity.

## Credentials

The plugin holds one Atagia service key in server memory. Provision a dedicated
key for the deployment; if a browser ever received a service key, do not reuse
it — provision a new one. See
[../PRODUCTION_CREDENTIALS.md](../PRODUCTION_CREDENTIALS.md). The boundary
supports exactly one current key and provides no fallback or dual-key window.

## Tests

Run the server-boundary tests from this directory:

```bash
npm test
```

Repository integration tests also exercise the official plugin shape, the
pinned host contract, multi-user mapping, spoof rejection, real Node-to-ASGI
authentication, credential rejection, and both-process restart behavior.

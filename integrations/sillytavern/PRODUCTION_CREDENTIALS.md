# SillyTavern Production Credentials

The Atagia Memory server plugin authenticates to Atagia with a single service
key held in the server process (or its secret manager). The browser extension
never holds or receives that key. This note covers provisioning that key for a
deployment, and replacing a key that was ever exposed to a browser or logs.

## Provisioning

- Generate a dedicated Atagia service key for the deployment and set it only in
  the server plugin's environment or secret manager (`ATAGIA_SERVICE_API_KEY`).
  Never place it in browser JavaScript, extension settings, local/session
  storage, HTML, or client network requests.
- Map each authenticated SillyTavern user to an Atagia user server-side via
  `ATAGIA_SILLYTAVERN_USER_MAP`. The browser cannot choose its Atagia user.
- The boundary accepts exactly one current key; there is no dual-key window.

## Replacing an exposed key

Older extension builds persisted the service key in browser settings, which
SillyTavern stores in plain text and other extensions can read. If a key was
ever exposed that way — or ended up in logs, screenshots, or a tool transcript
— do not reuse it in production. Generate a fresh key, set it in the server
secret store, update every server-side consumer of the old key, and confirm
Atagia no longer accepts the old value. Deleting the browser setting alone does
not account for copies already made elsewhere.

Because the boundary uses one key, replacing it is a coordinated
stop-and-restart of the plugin and any other consumer, not a live dual-key swap.

## Verification

- A request to Atagia with the old key returns `401`; the same-origin route
  works with the fresh server-held key, before and after restarting both Atagia
  and SillyTavern.
- The key does not appear in extension settings, local/session storage, the
  DOM, browser request headers/bodies, browser dev logs, or client bundles.
- Two mapped SillyTavern users each reach only their assigned Atagia user; a
  browser-supplied user ID, credential, or authorization field is rejected.
- With debug off, no user message, assistant response, system prompt, request
  payload, or memory preview is retained; reload and logout clear diagnostic
  metadata and the injected prompt.

Record only non-secret fingerprints and timestamps. Never write raw key values
to tickets, logs, shell history, screenshots, or this repository.

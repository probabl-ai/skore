---
status: accepted
date: 2026-10-15
decision-makers: ["@thomass-dev"]
consulted: ["@rouk1", "@glemaitre"]
---

# Authenticate Hub with API-keys instead of JWT/OAuth

## Context and Problem Statement

Hub access used to go through `skore.login` and a JWT/OAuth browser flow. That model
created friction in two places:

* **Environments.** Completing login required a browser (or equivalent interactive
  OAuth) dance that does not travel cleanly between a notebook, a terminal, CI, or a
  headless session. Tokens were short-lived and bound to that interactive path, so
  switching environment often meant logging in again.
* **Agents and subprocesses.** Coding agents and other child processes had to inherit
  or re-share the JWT through the environment. That is error-prone: the token expires,
  is process-local, and is easy for an agent to misuse (for example by inventing a
  `mode` argument on `skore.login`, see [#3261](https://github.com/probabl-ai/skore/issues/3261)).

We need a credential that is explicit, reusable across processes, and independent of
a browser session, while remaining scoped to a Hub host and workspace.

Related: [#3240](https://github.com/probabl-ai/skore/issues/3240),
[#3261](https://github.com/probabl-ai/skore/issues/3261).

## Decision Drivers

* Avoid browser/OAuth coupling so CLI, notebooks, CI, and agents use the same path.
* Let subprocesses and agents authenticate without passing a JWT/API-key through the
  environment.
* Keep credentials scoped to `(host, workspace)` rather than a single global token.
* Prefer a model users already know from git-style credential stores and tool CLIs
  (generate a key once, then reuse it).
* Drop the cross-plugin `skore.login(mode=...)` API, which was a no-op outside Hub
  and confused both humans and agents.

## Considered Options

* Keep JWT/OAuth via `skore.login`, optionally hiding `mode`.
* Authenticate only with an API-key from the environment (`SKORE_HUB_API_KEY`).
* Authenticate with a local API-key registry, with an environment-variable override.

## Decision Outcome

Chosen option: "Authenticate with a local API-key registry, with an environment-variable
override".

Hub project requests no longer attach a stored JWT. Outside Jupyterlite, which
authenticates with cookies, they send `X-API-Key`. Resolution order is:

1. `SKORE_HUB_API_KEY` if set,
2. otherwise the key for `(host, workspace)` in `~/.skore.hub/credentials.json`.

Users provision keys with the Hub CLI, not with `skore.login`:

```bash
$ skore hub api-key generate --host=<host> --workspace=<workspace>
$ skore hub api-key list
$ skore hub api-key revoke --host=<host> --workspace=<workspace>
```

`generate` and `revoke` still open a short-lived device OAuth login so the Hub can
mint or revoke the key. That token is not stored. Later Hub requests reuse the API
key.

The registry file is JSON. Storage mode is chosen when the file is created: `secret`
if a recommended [keyring](https://github.com/jaraco/keyring) backend is available,
otherwise `plaintext`. In `secret` mode the JSON file stores the key id, host, and
workspace, with `key` set to `null`; the secret itself lives in the system keyring.

```json
{
    "type": "plaintext",
    "keys": [
        {
            "id": 42,
            "host": "<host>",
            "workspace": "<workspace>",
            "key": "<key>"
        }
    ]
}
```

```json
{
    "type": "secret",
    "keys": [
        {
            "id": 42,
            "host": "<host>",
            "workspace": "<workspace>",
            "key": null
        }
    ]
}
```

The credentials directory is created with mode `0o700` and the JSON file with mode
`0o600`. Public `skore.login`, login plugins, and the stored JWT token module are
removed.

Key provisioning lives in this package (`skore hub api-key …`). An earlier iteration
depended on `skore-cli` and accepted the cycle `skore` → `skore-cli` → `skore`. That
cycle was broken, and the CLI then moved into `skore`.

### Consequences

* Good, because a key is a long-lived secret that can sit in a file, env var, or
  keyring and be reused by any process without a browser.
* Good, because agents do not need to share or refresh a JWT between processes; they
  read the same registry as the user.
* Good, because `skore.login` and its `mode` argument disappear, so Hub auth is only
  exercised when Hub is actually used.
* Good, because credentials are keyed by host and workspace, matching how projects are
  opened.
* Bad, because `generate` and `revoke` still need a browser for a short-lived device
  login. After that, Hub requests reuse the stored key.
* Bad, because a plaintext registry on disk is a stored secret; keyring mitigates this
  when a recommended backend exists, but is not guaranteed on every machine.
* Bad, because existing JWT-based scripts break (`skore.login` is gone). That is an
  intentional breaking change.

## More Information

Implemented in:
* https://github.com/probabl-ai/skore/pull/3255
* https://github.com/probabl-ai/skore-cli/pull/47
* https://github.com/probabl-ai/skore-cli/pull/48
* https://github.com/probabl-ai/skore-cli/pull/50

Follow-up, now part of this decision:
* https://github.com/probabl-ai/skore/pull/3308 broke the `skore` / `skore-cli` cycle.
* https://github.com/probabl-ai/skore/pull/3309 moved the CLI into the `skore` package.

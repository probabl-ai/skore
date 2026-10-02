---
status: accepted
date: 2026-09-22
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
  (generate or paste a key once, then reuse it).
* Drop the cross-plugin `skore.login(mode=...)` API, which was a no-op outside Hub
  and confused both humans and agents.

## Considered Options

* Keep JWT/OAuth via `skore.login`, optionally hiding `mode`.
* Authenticate only with an API-key from the environment (`SKORE_HUB_API_KEY`).
* Authenticate with a local API-key registry, with an environment-variable override.

## Decision Outcome

Chosen option: "Authenticate with a local API-key registry, with an environment-variable
override".

Hub clients no longer perform OAuth or attach a JWT. They send `X-API-Key`. Resolution
order is:

1. `SKORE_HUB_API_KEY` if set,
2. otherwise the key for `(host, workspace)` in `~/.skore.hub/credentials.json`.

Users provision keys with the Hub CLI, not with `skore.login`:

```bash
$ skore hub api-key generate --workspace=<workspace>
```

or

```bash
$ skore hub api-key add <api-key> --workspace=<workspace>
```

The registry file is JSON. Storage mode is chosen when the file is created: `secret`
if a recommended [keyring](https://github.com/jaraco/keyring) backend is available,
otherwise `plaintext`. In `secret` mode the JSON file stores only host and workspace;
the key itself lives in the system keyring.

```json
{
    "type": "plaintext",
    "keys": [
        {
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
            "host": "<host>",
            "workspace": "<workspace>"
        }
    ]
}
```

The credentials directory is created with mode `0o700` and the JSON file with mode
`0o600`. Public `skore.login`, login plugins, and the JWT token module are removed.

Key provisioning is implemented in `skore-cli` (`skore hub api-key …`). That CLI must
be a dependency of `skore` (Hub extra) so users get the commands with the library.
`skore-cli` already depends on `skore`, so this is a circular dependency
(`skore` → `skore-cli` → `skore`). We accept it for this iteration and will break the
cycle later.

### Consequences

* Good, because a key is a long-lived secret that can sit in a file, env var, or
  keyring and be reused by any process without a browser.
* Good, because agents do not need to share or refresh a JWT between processes; they
  read the same registry as the user.
* Good, because `skore.login` and its `mode` argument disappear, so Hub auth is only
  exercised when Hub is actually used.
* Good, because credentials are keyed by host and workspace, matching how projects are
  opened.
* Bad, because users must create or paste an API-key once (CLI), instead of a one-click
  browser login.
* Bad, because a plaintext registry on disk is a stored secret; keyring mitigates this
  when a recommended backend exists, but is not guaranteed on every machine.
* Bad, because existing JWT-based scripts break (`skore.login` is gone). That is an
  intentional breaking change.
* Bad, because adding `skore-cli` to `skore` dependencies creates a circular dependency
  that must be unwound in a later iteration.

## More Information

Implemented in:
* https://github.com/probabl-ai/skore/pull/3255
* https://github.com/probabl-ai/skore-cli/pull/47
* https://github.com/probabl-ai/skore-cli/pull/48
* https://github.com/probabl-ai/skore-cli/pull/50

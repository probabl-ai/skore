---
date: 2025-01-14
decision-makers: ["@augustebaum", "@thomass-dev"]
---

# Use `typing.Union` rather than `|` in type hints

## Context and Problem Statement

This ADR originates from a discussion in [this PR comment](https://github.com/probabl-ai/skore/pull/1084#discussion_r1914402895).

Writing a union type using `|` is [only available from Python 3.10](https://docs.python.org/3.12/library/typing.html#typing.Union), while we currently intend to support Python 3.9.
Even though we do not formally check type-hints using automatic tools like `mypy`, we still strive for consistency for style issues of this kind.

## Decision Outcome

Use `typing.Union` rather than `|` in type hints.

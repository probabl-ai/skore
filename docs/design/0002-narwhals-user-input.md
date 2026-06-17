---
status: "accepted"
date: 2026-06-17
decision-makers: @GaetandeCast
supersedes: 0001-postpone-narwhals
---

# Use narwhals for user input dataframe manipulation

## Context and Problem Statement

Skore accepts user `X` and `y` as pandas or polars DataFrames (and related array-likes).
Previously, manipulation of this data relied on private APIs from skrub (`skrub._dataframe` / `sbd`).

We want a stable, backend-agnostic layer for user-input dataframe operations without depending on skrub private APIs for that purpose.

## Decision Outcome

Chosen option: **Use [narwhals](https://narwhals-dev.github.io/narwhals/) (>=2.0.1) inline at call sites**

### Rules

1. **User input only**: use `nw.from_native()` / `.to_native()` when manipulating user-provided `X` / `y`.
2. **Skore outputs stay pandas**: metrics frames, check helpers, and hub serialization use pandas.
3. **Skrub reporting boundary**: convert to pandas with `nw.from_native(df).to_pandas()` before calling skrub reporting private APIs (`summarize_dataframe`, `to_html`).
4. **Sklearn boundary**: convert to pandas/numpy before `Pipeline.transform` when needed.
5. **No skore wrapper module**: call narwhals directly; normalization helpers live in `_utils/_dataframe.py` only.

### Consequences

* Good: removes skrub private dataframe dependency for user data paths.
* Good: pandas and polars supported uniformly; easier to extend later.
* Good: no skrub compatibility shim required (`tabular_pipeline` imported from public skrub API).
* Neutral: skrub reporting private APIs remain for TableReport HTML/summary generation.
* Bad: vertical concat must reset pandas index after `nw.concat(how="vertical")` to match prior `ignore_index=True` behaviour.

## Supersedes

[ADR 0001](0001-postpone-narwhals.md) — postponed narwhals adoption; this ADR reverses that decision for user-input manipulation only.

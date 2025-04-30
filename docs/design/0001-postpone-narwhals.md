---
# These are optional metadata elements. Feel free to remove any of them.
status: "accepted"
date: 2024-11-26
decision-makers: @augustebaum @tuscland
---

# Postone usage of generic DataFrame library

## Context and Problem Statement

skore projects natively support inserting `pandas` DataFrames, and with PR [#792](https://github.com/probabl-ai/skore/pull/792) it will also support `polars` DataFrames.
The question arises as to whether skore should use the `narwhals` library under the hood, or any another generic DataFrame library, as skore could then support many more DataFrame objects with little extra effort to integrate them.

## Considered Options

* Postpone usage of generic DataFrame library
* Use `narwhals`
* Use `arrow`

## Decision Outcome

Chosen option: "Postpone usage of generic DataFrame library"

<!-- This is an optional element. Feel free to remove. -->
### Consequences

* Good, because there is not enough foresight to know which generic library serves our needs the best
    - It is likely that we will explore generic formats in the near future in any case, for example:
        - If the storage backend for skore changes from `diskcache` to something else
        - if JSON as an exchange format is not robust enough for large data
    - Since we don't have experience with `narwhals` or `arrow`, integrating one of them might result in unexpected roadblocks which would take time to resolve
* Bad, because in the short term, adding support for a new DataFrame object will take more work
    - At the same time, it might be unlikely that support for another DataFrame implementation will be requested in the short term.

<!-- This is an optional element. Feel free to remove. -->
## More Information

This decision is being taken while reviewing PR [#792](https://github.com/probabl-ai/skore/pull/792), whose objective is to support inserting `polars` DataFrames.

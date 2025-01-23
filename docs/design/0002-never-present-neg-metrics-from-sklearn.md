---
status: accepted
date: 2025-01-06
decision-makers: ["@augustebaum", "@sylvaincom", "@glemaitre"]
consulted: ["@ogrisel"]
---

# Never show `neg_*` metrics from sklearn

## Context and Problem Statement

We show various metrics to users, many directly using sklearn.
In sklearn, many metrics are multiplied by -1 and prefixed with `neg_`, with the purpose of making all metrics "higher-is-better". This way, optimization tools in sklearn such as `GridSearchCV` do not need to figure out which way the metric should be optimized.
This is specific to sklearn, and there is no reason to port this design over to skore.

## Decision Drivers

* Our data-science-literate collaborators (@ogrisel, @glemaitre, @sylvaincom) consider the `neg_` trick should remain a solution to a sklearn-specific problem, and not be displayed in plots for the skore user.

## Decision Outcome

Chosen option: Never show `neg_*` metrics from sklearn in skore, only use the positive counterparts. This makes reports clearer.

### Consequences

* We show the most relevant information to the user.
* We might have to take on the responsibility of maintaining the "metric is higher-is-better" pre-condition ourselves.

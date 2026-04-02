.. _diagnostics:

===========
Diagnostics
===========

`skore` provides automated checks for common model quality pitfalls.
Use :meth:`~skore.EstimatorReport.diagnose` to run checks and get a summary of
detected issues.
Each issue has:

- a short explanation,
- a stable check code,
- and a link to this page.

Checks can be muted per call with `ignore=...`:

.. code-block:: python

    report.diagnose(ignore=["SKD001"])

You can also set a global ignore list with `configuration.ignore_checks = ...`:

.. code-block:: python

    from skore import configuration
    configuration.ignore_checks = ["SKD001"]

For cross-validation reports, checks are run per split and then aggregated
at report level through `~skore.CrossValidationReport.diagnose`. An issue is
reported only when it appears in a strict majority of evaluated splits.

For comparison reports, `~skore.ComparisonReport.diagnose` builds a global diagnostic
from each component report in the comparison. Issues are grouped by component
report and emitted as a single message.


.. _skd001-overfitting:

SKD001 - Potential overfitting
------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` compares train and test scores across the report's default predictive metrics
(timing metrics are excluded). A metric votes for overfitting when the train-favored
gap exceeds an adaptive threshold:

- **higher-is-better** metrics: ``train - test >= threshold``
- **lower-is-better** metrics: ``test - train >= threshold``

The threshold adapts to the scale of the scores:
``max(0.03, 0.10 * |reference|)`` where the reference is the train score for
higher-is-better metrics and the test score for lower-is-better metrics.
The floor of 0.03 prevents the threshold from vanishing on near-zero scores.

The check detects an issue when a **strict majority** of metrics vote for overfitting.

Why it matters
^^^^^^^^^^^^^^

A persistent train/test gap suggests the model has captured patterns specific to the
training data and may generalize poorly.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- simplify the model,
- regularize more strongly,
- improve feature engineering,
- use better validation protocols or more data.


.. _skd002-underfitting:

SKD002 - Potential underfitting
-------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` checks two conditions together across the report's default predictive metrics.
A metric votes for underfitting when **both** hold:

1. **Train and test scores are on par**: the absolute difference is within
   ``max(0.03, 0.05 * max(|train|, |test|))``.
2. **Neither score significantly outperforms a dummy baseline**: a score is considered
   significantly better than the baseline only when it exceeds
   ``max(0.01, 0.03 * |baseline|)``. The baseline is a ``DummyClassifier(strategy="prior")``
   for classification and a ``DummyRegressor(strategy="mean")`` for regression.

The check detects an issue when a **strict majority** of comparable metrics (those
present in both the estimator and dummy reports) vote for underfitting.

Why it matters
^^^^^^^^^^^^^^

When model performance is close to a naive baseline, the model is likely too simple,
under-trained, or using features that do not capture enough signal.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- increase model capacity,
- improve data representation and features,
- tune hyperparameters,
- collect richer data if possible.

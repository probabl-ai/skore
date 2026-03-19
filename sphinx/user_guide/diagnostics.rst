.. _diagnostics:

===========
Diagnostics
===========

`skore` diagnostics provide quick checks for common model quality pitfalls.
Use :meth:`EstimatorReport.diagnose` and :meth:`CrossValidationReport.diagnose`
to get concise findings, each with:

- a short explanation,
- a stable diagnostic code,
- and a link to this page.

Muting a diagnostic in v1 is done per call:

.. code-block:: python

    report.diagnose(ignore=["SKD001"])

You can also set a global ignore list:

.. code-block:: python

    from skore import config
    config.ignore_diagnostics = ("SKD001",)


Current coverage
----------------

- :class:`EstimatorReport`: full v1 diagnostics.
- :class:`CrossValidationReport`: split-level aggregation of the same diagnostics.
- :class:`ComparisonReport`: diagnostics are deferred for a later version.


Estimator and cross-validation behavior
---------------------------------------

For a single estimator report, diagnostics are computed directly from train/test metrics.
For a cross-validation report, diagnostics are computed per split and then aggregated
at report level; a diagnostic is reported as an issue only when it appears in a majority
of evaluated splits.


.. _skd001-overfitting:

SKD001 - Potential overfitting
------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` compares train and test scores across default predictive metrics
(time metrics are excluded). For each metric:

- if higher is better, a large train-vs-test drop votes toward overfitting;
- if lower is better, a large test-vs-train increase votes toward overfitting.

The diagnostic is flagged when a majority of comparable metrics vote for overfitting.

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

`skore` checks two signals together across default predictive metrics:

- train and test scores are on par,
- and both are not significantly better than a dummy baseline.

The diagnostic is flagged when a majority of comparable metrics satisfy both conditions.

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


.. _comparison-report-diagnostics:

ComparisonReport diagnostics
----------------------------

Comparison-level diagnostics are intentionally deferred in this version.
For now, call :meth:`diagnose` on each component estimator or cross-validation report.
Later versions will aggregate those diagnostics at comparison level.

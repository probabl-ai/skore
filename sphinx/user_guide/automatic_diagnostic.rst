.. _automatic_diagnostic:

====================
Automatic diagnostic
====================

`skore` provides automated checks for common model quality pitfalls.
Use :meth:`~skore.EstimatorReport.diagnose` to run checks and get a diagnostic that
summarizes detected issues.
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


.. _skd003-inconsistent_performance:

SKD003 - Inconsistent performance across folds
-----------------------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

This check applies only to :class:`~skore.CrossValidationReport`.

`skore` examines each fold's test scores across the report's default predictive metrics
(timing metrics are excluded). For every metric, a **modified Z-score** based on the
Median Absolute Deviation (MAD) is computed:

.. math::

   z_i = 0.6745 \times \frac{x_i - \widetilde{x}}{\text{MAD}}

A fold is flagged as an outlier for a given metric when :math:`|z_i| > 3.5`.

A fold votes as inconsistent when a **strict majority** of metrics flag it as an
outlier. The check detects an issue when at least one fold is voted inconsistent.

Why it matters
^^^^^^^^^^^^^^

When one or more folds perform very differently from the rest, the cross-validation
estimate becomes unreliable. The anomaly may reveal data leakage, uneven class
distributions across folds, or a model that is sensitive to specific data splits.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- use stratified or grouped cross-validation to ensure a more even split,
- investigate whether the outlier fold contains a different data distribution,
- check for data leakage or temporal effects,
- increase the size of the dataset to improve stability.


.. _skd004-high_class_imbalance:

SKD004 - High class imbalance
------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

This check applies only to classification tasks.

`skore` counts the occurrences of each class across the train and test sets. The check
detects an issue when the most frequent class represents more than **80 %** of the
dataset.

Why it matters
^^^^^^^^^^^^^^

When one class dominates the dataset, a model can achieve high accuracy simply by
predicting the majority class. Accuracy alone becomes a misleading performance
indicator, and the model may fail to detect the minority class entirely.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- use metrics that account for imbalance (precision, recall, F1, ROC AUC),
- resample the dataset (oversampling the minority or undersampling the majority),
- use class weights in the estimator,
- collect more data for the minority class if possible.

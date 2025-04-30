Report for a comparison of :class:`EstimatorReport`
===================================================

.. currentmodule:: skore

The class :class:`ComparisonReport` provides a report allowing to compare :class:`EstimatorReport` instances in an interactive way. The functionalities of the report are accessible through accessors.

Main class
----------

.. autosummary::
   :toctree: ../api/
   :nosignatures:
   :template: base.rst

Methods
-------

.. rubric:: Methods

.. autoclass:: ComparisonReport
   :members: help, cache_predictions, clear_cache, get_predictions
   :exclude-members: metrics
   :noindex:

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of the compared estimators. In addition, we provide a sub-accessor `plot`, to get the common performance metric representations.

.. rubric:: Metrics

.. autosummary::
   :toctree: ../api/
   :nosignatures:
   :template: autosummary/accessor_method.rst

    ComparisonReport.metrics.help
    ComparisonReport.metrics.report_metrics
    ComparisonReport.metrics.custom_metric
    ComparisonReport.metrics.accuracy
    ComparisonReport.metrics.brier_score
    ComparisonReport.metrics.log_loss
    ComparisonReport.metrics.precision
    ComparisonReport.metrics.r2
    ComparisonReport.metrics.recall
    ComparisonReport.metrics.rmse
    ComparisonReport.metrics.roc_auc
    ComparisonReport.metrics.timings

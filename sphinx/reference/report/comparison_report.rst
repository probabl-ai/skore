Report for a comparison of :class:`EstimatorReport`
===================================================

.. currentmodule:: skore

The class :class:`ComparisonReport` provides a report allowing to compare
:class:`EstimatorReport` instances in an interactive way. The functionalities of the
report are accessible through accessors.

.. autosummary::
    :toctree: ../api/
    :template: class_with_accessors.rst

    ComparisonReport

.. rubric:: Methods

.. autosummary::
    :toctree: ../api/
    :template: class_methods_no_index.rst

    ComparisonReport.help
    ComparisonReport.cache_predictions
    ComparisonReport.clear_cache
    ComparisonReport.get_predictions

.. rubric:: Accessors

.. autosummary::
    :toctree: ../api/
    :nosignatures:
    :template: autosummary/accessor.rst

    ComparisonReport.metrics

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of the
compared estimators. In addition, we provide a sub-accessor `plot`, to
get the common performance metric representations.

.. autosummary::
    :toctree: ../api/
    :template: autosummary/accessor_method.rst

    ComparisonReport.metrics.help
    ComparisonReport.metrics.summarize
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

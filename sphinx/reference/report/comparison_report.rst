Comparing multiple reports
==========================

.. currentmodule:: skore

The class :class:`ComparisonReport` provides a report allowing to compare
:class:`EstimatorReport` or :class:`CrossValidationReport` instances in an interactive
way. The functionalities of the report are accessible through accessors.

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
    ComparisonReport.create_estimator_report
    ComparisonReport.get_predictions

.. rubric:: Accessors

.. autosummary::
    :toctree: ../api/
    :nosignatures:
    :template: autosummary/accessor.rst

    ComparisonReport.inspection
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

    ComparisonReport.metrics.accuracy
    ComparisonReport.metrics.brier_score
    ComparisonReport.metrics.confusion_matrix
    ComparisonReport.metrics.custom_metric
    ComparisonReport.metrics.log_loss
    ComparisonReport.metrics.precision
    ComparisonReport.metrics.precision_recall
    ComparisonReport.metrics.prediction_error
    ComparisonReport.metrics.r2
    ComparisonReport.metrics.recall
    ComparisonReport.metrics.rmse
    ComparisonReport.metrics.roc
    ComparisonReport.metrics.roc_auc
    ComparisonReport.metrics.timings

Inspection
----------

The `inspection` accessor helps you inspect your model by e.g. evaluating the importance
of the features in your model.

.. autosummary::
    :toctree: ../api/
    :template: autosummary/accessor_method.rst

    ComparisonReport.inspection.help
    ComparisonReport.inspection.coefficients
    ComparisonReport.inspection.impurity_decrease

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

.. include:: ../api/ComparisonReport.metrics.inc

Inspection
----------

The `inspection` accessor helps you inspect your model by e.g. evaluating the importance
of the features in your model.

.. include:: ../api/ComparisonReport.inspection.inc

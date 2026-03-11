Report for a cross-validation of an estimator
=============================================

.. currentmodule:: skore

The class :class:`CrossValidationReport` performs cross-validation and provides a report
to inspect and evaluate a scikit-learn estimator in an interactive way. The
functionalities of the report are exposed through accessors.

.. autosummary::
   :toctree: ../api/
   :template: class_with_accessors.rst

   CrossValidationReport

.. rubric:: Methods

.. autosummary::
   :toctree: ../api/
   :template: class_methods_no_index.rst

   CrossValidationReport.help
   CrossValidationReport.cache_predictions
   CrossValidationReport.clear_cache
   CrossValidationReport.create_estimator_report
   CrossValidationReport.get_predictions

.. rubric:: Accessors

.. autosummary::
   :toctree: ../api/
   :template: autosummary/accessor.rst

   CrossValidationReport.data
   CrossValidationReport.inspection
   CrossValidationReport.metrics


.. _cross_validation_data:

Data
----

The `data` accessor helps you to get insights about the dataset used in the
cross-validation.

.. include:: ../api/CrossValidationReport.data.inc


.. _cross_validation_metrics:

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of your
estimator across cross-validation splits.

.. include:: ../api/CrossValidationReport.metrics.inc

Inspection
----------

The `inspection` accessor helps you inspect your model by e.g. evaluating the importance
of the features in your model.

.. include:: ../api/CrossValidationReport.inspection.inc

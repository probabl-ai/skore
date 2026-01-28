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
   CrossValidationReport.feature_importance
   CrossValidationReport.metrics


.. _cross_validation_data:

Data
----

The `data` accessor helps you to get insights about the dataset used in the
cross-validation.

.. autosummary::
    :toctree: ../api/
    :template: autosummary/accessor_method.rst

    CrossValidationReport.data.help
    CrossValidationReport.data.analyze


.. _cross_validation_metrics:

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of your
estimator across cross-validation splits.

.. autosummary::
    :toctree: ../api/
    :template: autosummary/accessor_method.rst

    CrossValidationReport.metrics.help
    CrossValidationReport.metrics.summarize
    CrossValidationReport.metrics.custom_metric
    CrossValidationReport.metrics.accuracy
    CrossValidationReport.metrics.brier_score
    CrossValidationReport.metrics.log_loss
    CrossValidationReport.metrics.precision
    CrossValidationReport.metrics.precision_recall
    CrossValidationReport.metrics.prediction_error
    CrossValidationReport.metrics.r2
    CrossValidationReport.metrics.recall
    CrossValidationReport.metrics.rmse
    CrossValidationReport.metrics.roc
    CrossValidationReport.metrics.roc_auc
    CrossValidationReport.metrics.timings

Feature importance
------------------

The `feature_importance` accessor helps you evaluate the importance
used to train your estimator.

.. autosummary::
    :toctree: ../api/
    :template: autosummary/accessor_method.rst

    CrossValidationReport.feature_importance.help
    CrossValidationReport.feature_importance.coefficients

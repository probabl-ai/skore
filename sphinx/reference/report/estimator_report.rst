Report for a single estimator
=============================

.. currentmodule:: skore

The class :class:`EstimatorReport` provides a report allowing to inspect and
evaluate a scikit-learn estimator in an interactive way. The functionalities of the
report are accessible through accessors.

.. autosummary::
   :toctree: ../api/
   :template: base.rst

   EstimatorReport

.. rubric:: Methods

.. autosummary::
   :toctree: ../api/
   :template: class_methods_no_index.rst

   EstimatorReport.help
   EstimatorReport.cache_predictions
   EstimatorReport.clear_cache
   EstimatorReport.get_predictions

.. rubric:: Metrics

.. autosummary::
   :toctree: ../api/
   :template: autosummary/accessor.rst

   EstimatorReport.metrics

.. rubric:: Feature importance

.. autosummary::
   :toctree: ../api/
   :template: autosummary/accessor.rst

   EstimatorReport.feature_importance

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of your
estimator.

.. autosummary::
    :toctree: ../api/
    :template: autosummary/accessor_method.rst

    EstimatorReport.metrics.help
    EstimatorReport.metrics.report_metrics
    EstimatorReport.metrics.custom_metric
    EstimatorReport.metrics.timings
    EstimatorReport.metrics.accuracy
    EstimatorReport.metrics.brier_score
    EstimatorReport.metrics.log_loss
    EstimatorReport.metrics.precision
    EstimatorReport.metrics.precision_recall
    EstimatorReport.metrics.prediction_error
    EstimatorReport.metrics.r2
    EstimatorReport.metrics.recall
    EstimatorReport.metrics.rmse
    EstimatorReport.metrics.roc
    EstimatorReport.metrics.roc_auc

Feature importance
------------------

The `feature_importance` accessor helps you to evaluate the importance of the features
used to train your estimator.

.. autosummary::
    :toctree: ../api/
    :template: autosummary/accessor_method.rst

    EstimatorReport.feature_importance.help
    EstimatorReport.feature_importance.coefficients
    EstimatorReport.feature_importance.mean_decrease_impurity
    EstimatorReport.feature_importance.permutation

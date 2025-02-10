Report for a single estimator
============================

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
   :template: autosummary/accessor_method.rst

   EstimatorReport.help

.. rubric:: Metrics

.. autosummary::
   :toctree: ../api/
   :template: autosummary/accessor.rst

   EstimatorReport.metrics

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of your
estimator.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: autosummary/accessor_method.rst

    EstimatorReport.metrics.help
    EstimatorReport.metrics.report_metrics
    EstimatorReport.metrics.custom_metric
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

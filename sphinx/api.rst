API
===

.. currentmodule:: skore

This page lists all the public functions and classes of the skore package.

.. warning ::

    This code is still in development. **The API is subject to change.**

Project
-------

These functions and classes are meant for managing a Project.

.. autosummary::
    :toctree: generated/
    :template: base.rst
    :caption: Managing a project

    Project
    Project.put
    Project.get

Get assistance when developing ML/DS projects
---------------------------------------------

These functions and classes enhance scikit-learn's ones.

.. autosummary::
    :toctree: generated/
    :template: base.rst
    :caption: ML Assistance

    train_test_split

Report for a single estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class :class:`EstimatorReport` provides a report allowing to inspect and
evaluate a scikit-learn estimator in an interactive way. The functionalities of the
report are accessible through accessors.

.. autosummary::
    :toctree: generated/
    :template: base.rst

    EstimatorReport

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: autosummary/accessor_method.rst

    EstimatorReport.help

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: autosummary/accessor.rst

    EstimatorReport.metrics

Metrics
"""""""

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


Cross-validation report for an estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class :class:`CrossValidationReport` provides a report allowing to inspect and
evaluate a scikit-learn estimator through cross-validation in an interactive way. The
functionalities of the report are accessible through accessors.

.. autosummary::
    :toctree: generated/
    :template: base.rst

    CrossValidationReport

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: autosummary/accessor_method.rst

    CrossValidationReport.help

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: autosummary/accessor.rst

    CrossValidationReport.metrics

Metrics
"""""""

The `metrics` accessor helps you to evaluate the statistical performance of your
estimator during a cross-validation.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: autosummary/accessor_method.rst

    CrossValidationReport.metrics.help
    CrossValidationReport.metrics.report_metrics
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

.. Deprecated
.. ----------

.. These functions and classes are deprecated.

.. .. autosummary::
..     :toctree: generated/
..     :template: base.rst
..     :caption: Deprecated
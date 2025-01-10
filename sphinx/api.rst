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
    item.primitive_item.PrimitiveItem
    open

Get assistance when developing ML/DS projects
---------------------------------------------

These functions and classes enhance scikit-learn's ones.

.. autosummary::
    :toctree: generated/
    :template: base.rst
    :caption: ML Assistance

    train_test_split
    CrossValidationReporter
    item.cross_validation_item.CrossValidationItem

Report for a single estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class :class:`EstimatorReport` provides a reporter allowing to inspect and
evaluate a scikit-learn estimator in an interactive way. The functionalities of the
reporter are accessible through accessors.

.. autosummary::
    :toctree: generated/
    :template: base.rst
    :caption: Reporting for a single estimator

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
estimator. In addition, we provide a sub-accessor `plot`, to get the common
performance metric representations.

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
    EstimatorReport.metrics.r2
    EstimatorReport.metrics.recall
    EstimatorReport.metrics.rmse
    EstimatorReport.metrics.roc_auc
    EstimatorReport.metrics.plot.help
    EstimatorReport.metrics.plot.precision_recall
    EstimatorReport.metrics.plot.prediction_error
    EstimatorReport.metrics.plot.roc

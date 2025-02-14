ML Assistance
=============

.. currentmodule:: skore

This section contains documentation for skore features that enhance the ML development process.

Get assistance when developing ML/DS projects
---------------------------------------------

These functions and classes build upon scikit-learn's functionality.

.. autosummary::
    :toctree: ../api/
    :template: base.rst

    train_test_split

Single Estimator Report
-----------------------

:class:`skore.EstimatorReport` provides comprehensive reporting capabilities for individual
scikit-learn estimators, including metrics, visualizations, and evaluation tools.

.. toctree::
   :maxdepth: 2
   :hidden:

   estimator_report

Cross-validation Report
-----------------------

:class:`skore.CrossValidationReport` provides comprehensive capabilities for evaluating scikit-learn estimators by cross-validation, and reporting the results.

.. toctree::
   :maxdepth: 2
   :hidden:

   cross_validation_report

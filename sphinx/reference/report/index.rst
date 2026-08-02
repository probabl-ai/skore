ML Assistance
=============

.. currentmodule:: skore

This section contains documentation for skore features that enhance the ML development
process.

Get assistance when developing ML/DS projects
---------------------------------------------

:func:`evaluate` is the main entry point: pass one or more estimators and data to get
an :class:`EstimatorReport`, :class:`CrossValidationReport`, or
:class:`ComparisonReport`. The report classes remain available for advanced use, but
:func:`evaluate` is usually the simpler way to create them.

These functions and classes build upon scikit-learn's functionality.

.. autosummary::
    :toctree: ../api/
    :template: base.rst

    evaluate
    compare
    TrainTestSplit

Single Estimator Report
-----------------------

:class:`skore.EstimatorReport` provides comprehensive reporting capabilities for
individual scikit-learn estimators, including metrics, visualizations, and evaluation
tools. Prefer creating it with :func:`evaluate`.

.. toctree::
   :maxdepth: 2
   :hidden:

   estimator_report

Cross-validation Report
-----------------------

:class:`skore.CrossValidationReport` provides comprehensive capabilities for evaluating
scikit-learn estimators by cross-validation, and reporting the results. Prefer creating
it with :func:`evaluate` and an integer or CV ``splitter``.

.. toctree::
   :maxdepth: 2
   :hidden:

   cross_validation_report

Comparison Report
-----------------

:class:`skore.ComparisonReport` provides comprehensive capabilities for comparing
:class:`skore.EstimatorReport` or :class:`skore.CrossValidationReport` instances, and
reporting the results. Prefer creating it with :func:`evaluate` (several estimators) or
:func:`compare` (existing reports).

.. toctree::
   :maxdepth: 2
   :hidden:

   comparison_report

Visualization Displays
----------------------

A set of displays are available through the different reports. Find in this section
the API of each display.

.. toctree::
   :maxdepth: 2
   :hidden:

   displays

Checks
------

Checks classes used by the ``checks`` accessor on reports.

.. autosummary::
    :toctree: ../api/
    :template: base.rst

    ChecksSummaryDisplay
    Check
    CheckNotApplicable

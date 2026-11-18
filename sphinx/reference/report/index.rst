ML Assistance
=============

.. currentmodule:: skore

This section contains documentation for skore features that enhance the ML development
process.

Creating reports
----------------

:func:`evaluate` and :func:`compare` are the entry points for building reports.

.. toctree::
   :maxdepth: 2
   :hidden:

   creating_reports

Estimator report
----------------

:class:`skore.EstimatorReport` provides comprehensive reporting capabilities for
individual scikit-learn estimators, including metrics, visualizations, and evaluation
tools.

.. toctree::
   :maxdepth: 2
   :hidden:

   estimator_report

Cross-validation report
-----------------------

:class:`skore.CrossValidationReport` provides comprehensive capabilities for evaluating
scikit-learn estimators by cross-validation, and reporting the results.

.. toctree::
   :maxdepth: 2
   :hidden:

   cross_validation_report

Comparison report
-----------------

:class:`skore.ComparisonReport` provides comprehensive capabilities for comparing
:class:`skore.EstimatorReport` or :class:`skore.CrossValidationReport` instances, and
reporting the results.

.. toctree::
   :maxdepth: 2
   :hidden:

   comparison_report

Visualization displays
----------------------

A set of displays are available through the different reports. Find in this section
the API of each display.

.. toctree::
   :maxdepth: 2
   :hidden:

   displays

Methodological checks
---------------------

Checks classes used by the ``checks`` accessor on reports.

.. toctree::
   :maxdepth: 2
   :hidden:

   checks

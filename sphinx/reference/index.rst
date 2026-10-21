API
===

This page lists all the public functions and classes of the skore package.

.. warning ::

    This code is still in development. **The API is subject to change.**

.. currentmodule:: skore

Overview
--------

The following table provides a quick reference to the public classes and functions in skore:

Project Management
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :class:`Project`
     - Main class for managing a skore project and its reports
   * - :class:`Summary`
     - Summary of the reports stored in a project
   * - :class:`Experiments`
     - The experiments stored in a project
   * - :class:`Experiment`
     - One experiment and the methods that move it


ML Assistance
^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :func:`evaluate`
     - Evaluate one or more estimators on the given data.
   * - :func:`compare`
     - Compare pre-existing reports with a :class:`~skore.ComparisonReport`
   * - :class:`TrainTestSplit`
     - Single train-test splitter compatible with the `scikit-learn CV splitter
       API <https://scikit-learn.org/stable/glossary.html#term-CV-splitter>`_

.. include:: api/accessor_tables.rst


Display Classes
^^^^^^^^^^^^^^^

Data
""""

.. list-table::
   :widths: 30 70

   * - :class:`TableReportDisplay`
     - Display for tabular data reports


Metrics
"""""""

.. list-table::
   :widths: 30 70

   * - :class:`MetricsSummaryDisplay`
     - Display for summarizing multiple metrics
   * - :class:`ConfusionMatrixDisplay`
     - Confusion matrix visualization
   * - :class:`RocCurveDisplay`
     - ROC (Receiver Operating Characteristic) curve visualization
   * - :class:`PrecisionRecallCurveDisplay`
     - Precision-Recall curve visualization
   * - :class:`PredictionErrorDisplay`
     - Prediction error visualization


Inspection
""""""""""

.. list-table::
   :widths: 30 70

   * - :class:`CoefficientsDisplay`
     - Display for visualizing feature importance via model coefficients
   * - :class:`ImpurityDecreaseDisplay`
     - Display for visualizing feature importance via Mean Decrease in Impurity (MDI)
   * - :class:`PermutationImportanceDisplay`
     - Display for visualizing feature importance via permutation importance
   * - :class:`CalibrationDisplay`
     - Display for visualizing classifier calibration curves


General
"""""""

.. list-table::
   :widths: 30 70

   * - :class:`Display`
     - Protocol implemented by every visualization display


Checks
^^^^^^

.. list-table::
   :widths: 30 70

   * - :class:`ChecksSummaryDisplay`
     - Display for check results returned by the ``checks`` accessor
   * - :class:`Check`
     - Protocol for defining custom checks
   * - :exc:`CheckNotApplicable`
     - Exception raised when a check cannot run on the given report


Utilities
^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :obj:`configuration`
     - Global configuration for `skore` also usable as a context manager
   * - :func:`show_versions`
     - Print version information for skore and its dependencies


Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :ref:`skore <cli-reference>`
     - Synchronize projects, install agent skills, and connect a coding agent to Skore Hub


.. toctree::
   :maxdepth: 2
   :hidden:

   Managing a project <project>
   ML Assistance <report/index>
   Utilities <utils>
   Command-line interface <cli/index>

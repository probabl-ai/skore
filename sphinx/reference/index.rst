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


ML Assistance
^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :func:`train_test_split`
     - Split arrays or matrices into random train and test subsets

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


Utilities
^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :obj:`configuration`
     - Global configuration for `skore` also usable as a context manager
   * - :func:`show_versions`
     - Print version information for skore and its dependencies


.. toctree::
   :maxdepth: 2
   :hidden:

   Managing a project <project>
   ML Assistance <report/index>
   Utilities <utils>

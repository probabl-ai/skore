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
     - Display for visualizing model coefficients
   * - :class:`PermutationImportanceDisplay`
     - Display for permutation feature importance


Configuration
^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :func:`config_context`
     - Context manager for temporarily modifying skore configuration
   * - :func:`get_config`
     - Get current skore configuration values
   * - :func:`set_config`
     - Set skore configuration values


Utilities
^^^^^^^^^

.. list-table::
   :widths: 30 70

   * - :func:`show_versions`
     - Print version information for skore and its dependencies


.. toctree::
   :maxdepth: 2
   :hidden:

   Managing a project <project>
   ML Assistance <report/index>
   Utilities <utils>

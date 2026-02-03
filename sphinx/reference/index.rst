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

   * - :class:`EstimatorReport`
     - Comprehensive report for evaluating a single scikit-learn estimator
   * - :class:`CrossValidationReport`
     - Report for evaluating an estimator using cross-validation
   * - :class:`ComparisonReport`
     - Report for comparing multiple estimator or cross-validation reports
   * - :func:`train_test_split`
     - Split arrays or matrices into random train and test subsets

.. dropdown:: EstimatorReport accessors
   :icon: chevron-down

   **Data**: :class:`EstimatorReport.data`

   .. list-table::
      :widths: 30 70

      * - :func:`~EstimatorReport.data.analyze`
        - Plot dataset statistics

   **Metrics**: :class:`EstimatorReport.metrics`

   .. list-table::
      :widths: 30 70

      * - :func:`~EstimatorReport.metrics.summarize`
        - Report a set of metrics for the estimator
      * - :func:`~EstimatorReport.metrics.timings`
        - Get all measured processing times
      * - :func:`~EstimatorReport.metrics.accuracy`
        - Compute accuracy score
      * - :func:`~EstimatorReport.metrics.precision`
        - Compute precision score
      * - :func:`~EstimatorReport.metrics.recall`
        - Compute recall score
      * - :func:`~EstimatorReport.metrics.brier_score`
        - Compute Brier score
      * - :func:`~EstimatorReport.metrics.roc_auc`
        - Compute ROC AUC score
      * - :func:`~EstimatorReport.metrics.log_loss`
        - Compute log loss
      * - :func:`~EstimatorReport.metrics.r2`
        - Compute R² score
      * - :func:`~EstimatorReport.metrics.rmse`
        - Compute root mean squared error
      * - :func:`~EstimatorReport.metrics.custom_metric`
        - Compute a custom metric
      * - :func:`~EstimatorReport.metrics.confusion_matrix`
        - Plot confusion matrix
      * - :func:`~EstimatorReport.metrics.roc`
        - Plot ROC curve
      * - :func:`~EstimatorReport.metrics.precision_recall`
        - Plot precision-recall curve
      * - :func:`~EstimatorReport.metrics.prediction_error`
        - Plot prediction error

   **Inspection**: :class:`EstimatorReport.inspection`

   .. list-table::
      :widths: 30 70

      * - :func:`~EstimatorReport.inspection.coefficients`
        - Retrieve coefficients of a linear model
      * - :func:`~EstimatorReport.inspection.impurity_decrease`
        - Retrieve mean decrease impurity of tree-based models
      * - :func:`~EstimatorReport.inspection.permutation_importance`
        - Report permutation feature importance

.. dropdown:: CrossValidationReport accessors
   :icon: chevron-down

   **Data**: :class:`CrossValidationReport.data`

   .. list-table::
      :widths: 30 70

      * - :func:`~CrossValidationReport.data.analyze`
        - Plot dataset statistics

   **Metrics**: :class:`CrossValidationReport.metrics`

   .. list-table::
      :widths: 30 70

      * - :func:`~CrossValidationReport.metrics.summarize`
        - Report a set of metrics with aggregation options
      * - :func:`~CrossValidationReport.metrics.timings`
        - Get measured processing times across CV splits
      * - :func:`~CrossValidationReport.metrics.accuracy`
        - Compute accuracy score across splits
      * - :func:`~CrossValidationReport.metrics.precision`
        - Compute precision score across splits
      * - :func:`~CrossValidationReport.metrics.recall`
        - Compute recall score across splits
      * - :func:`~CrossValidationReport.metrics.brier_score`
        - Compute Brier score across splits
      * - :func:`~CrossValidationReport.metrics.roc_auc`
        - Compute ROC AUC score across splits
      * - :func:`~CrossValidationReport.metrics.log_loss`
        - Compute log loss across splits
      * - :func:`~CrossValidationReport.metrics.r2`
        - Compute R² score across splits
      * - :func:`~CrossValidationReport.metrics.rmse`
        - Compute RMSE across splits
      * - :func:`~CrossValidationReport.metrics.custom_metric`
        - Compute custom metrics across splits
      * - :func:`~CrossValidationReport.metrics.confusion_matrix`
        - Plot confusion matrices for CV splits
      * - :func:`~CrossValidationReport.metrics.roc`
        - Plot ROC curves for CV splits
      * - :func:`~CrossValidationReport.metrics.precision_recall`
        - Plot precision-recall curves for CV splits
      * - :func:`~CrossValidationReport.metrics.prediction_error`
        - Plot prediction errors for CV splits

   **Inspection**: :class:`CrossValidationReport.inspection`

   .. list-table::
      :widths: 30 70

      * - :func:`~CrossValidationReport.inspection.coefficients`
        - Retrieve coefficients across CV splits

.. dropdown:: ComparisonReport accessors
   :icon: chevron-down

   **Metrics**: :class:`ComparisonReport.metrics`

   .. list-table::
      :widths: 30 70

      * - :func:`~ComparisonReport.metrics.summarize`
        - Report metrics comparing multiple estimators
      * - :func:`~ComparisonReport.metrics.timings`
        - Get timings across compared estimators
      * - :func:`~ComparisonReport.metrics.accuracy`
        - Compare accuracy scores
      * - :func:`~ComparisonReport.metrics.precision`
        - Compare precision scores
      * - :func:`~ComparisonReport.metrics.recall`
        - Compare recall scores
      * - :func:`~ComparisonReport.metrics.brier_score`
        - Compare Brier scores
      * - :func:`~ComparisonReport.metrics.roc_auc`
        - Compare ROC AUC scores
      * - :func:`~ComparisonReport.metrics.log_loss`
        - Compare log loss
      * - :func:`~ComparisonReport.metrics.r2`
        - Compare R² scores
      * - :func:`~ComparisonReport.metrics.rmse`
        - Compare RMSE
      * - :func:`~ComparisonReport.metrics.custom_metric`
        - Compare custom metrics
      * - :func:`~ComparisonReport.metrics.confusion_matrix`
        - Plot confusion matrices for comparison
      * - :func:`~ComparisonReport.metrics.roc`
        - Plot ROC curves for comparison
      * - :func:`~ComparisonReport.metrics.precision_recall`
        - Plot precision-recall curves for comparison
      * - :func:`~ComparisonReport.metrics.prediction_error`
        - Plot prediction errors for comparison

   **Inspection**: :class:`ComparisonReport.inspection`

   .. list-table::
      :widths: 30 70

      * - :func:`~ComparisonReport.inspection.coefficients`
        - Compare coefficients across estimators


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

   ML Assistance <report/index>
   Managing a project <project>
   Utilities <utils>

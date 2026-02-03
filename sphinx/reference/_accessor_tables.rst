
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

      * - :func:`~EstimatorReport.metrics.accuracy`
        - Compute the accuracy score

      * - :func:`~EstimatorReport.metrics.brier_score`
        - Compute the Brier score

      * - :func:`~EstimatorReport.metrics.confusion_matrix`
        - Plot the confusion matrix

      * - :func:`~EstimatorReport.metrics.custom_metric`
        - Compute a custom metric

      * - :func:`~EstimatorReport.metrics.log_loss`
        - Compute the log loss

      * - :func:`~EstimatorReport.metrics.precision`
        - Compute the precision score

      * - :func:`~EstimatorReport.metrics.precision_recall`
        - Plot the precision-recall curve

      * - :func:`~EstimatorReport.metrics.prediction_error`
        - Plot the prediction error of a regression model

      * - :func:`~EstimatorReport.metrics.r2`
        - Compute the R² score

      * - :func:`~EstimatorReport.metrics.recall`
        - Compute the recall score

      * - :func:`~EstimatorReport.metrics.rmse`
        - Compute the root mean squared error

      * - :func:`~EstimatorReport.metrics.roc`
        - Plot the ROC curve

      * - :func:`~EstimatorReport.metrics.roc_auc`
        - Compute the ROC AUC score

      * - :func:`~EstimatorReport.metrics.summarize`
        - Report a set of metrics for our estimator

      * - :func:`~EstimatorReport.metrics.timings`
        - Get all measured processing times related to the estimator


   **Inspection**: :class:`EstimatorReport.inspection`

   .. list-table::
      :widths: 30 70

      * - :func:`~EstimatorReport.inspection.coefficients`
        - Retrieve the coefficients of a linear model, including the intercept

      * - :func:`~EstimatorReport.inspection.impurity_decrease`
        - Retrieve the mean decrease impurity (MDI) of a tree-based model

      * - :func:`~EstimatorReport.inspection.permutation_importance`
        - Report the permutation feature importance




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

      * - :func:`~CrossValidationReport.metrics.accuracy`
        - Compute the accuracy score

      * - :func:`~CrossValidationReport.metrics.brier_score`
        - Compute the Brier score

      * - :func:`~CrossValidationReport.metrics.confusion_matrix`
        - Plot the confusion matrix

      * - :func:`~CrossValidationReport.metrics.custom_metric`
        - Compute a custom metric

      * - :func:`~CrossValidationReport.metrics.log_loss`
        - Compute the log loss

      * - :func:`~CrossValidationReport.metrics.precision`
        - Compute the precision score

      * - :func:`~CrossValidationReport.metrics.precision_recall`
        - Plot the precision-recall curve

      * - :func:`~CrossValidationReport.metrics.prediction_error`
        - Plot the prediction error of a regression model

      * - :func:`~CrossValidationReport.metrics.r2`
        - Compute the R² score

      * - :func:`~CrossValidationReport.metrics.recall`
        - Compute the recall score

      * - :func:`~CrossValidationReport.metrics.rmse`
        - Compute the root mean squared error

      * - :func:`~CrossValidationReport.metrics.roc`
        - Plot the ROC curve

      * - :func:`~CrossValidationReport.metrics.roc_auc`
        - Compute the ROC AUC score

      * - :func:`~CrossValidationReport.metrics.summarize`
        - Report a set of metrics for our estimator

      * - :func:`~CrossValidationReport.metrics.timings`
        - Get all measured processing times related to the estimator


   **Inspection**: :class:`CrossValidationReport.inspection`

   .. list-table::
      :widths: 30 70

      * - :func:`~CrossValidationReport.inspection.coefficients`
        - Retrieve the coefficients across splits, including the intercept




.. dropdown:: ComparisonReport accessors
   :icon: chevron-down

   **Metrics**: :class:`ComparisonReport.metrics`

   .. list-table::
      :widths: 30 70

      * - :func:`~ComparisonReport.metrics.accuracy`
        - Compute the accuracy score

      * - :func:`~ComparisonReport.metrics.brier_score`
        - Compute the Brier score

      * - :func:`~ComparisonReport.metrics.confusion_matrix`
        - Plot the confusion matrix

      * - :func:`~ComparisonReport.metrics.custom_metric`
        - Compute a custom metric

      * - :func:`~ComparisonReport.metrics.log_loss`
        - Compute the log loss

      * - :func:`~ComparisonReport.metrics.precision`
        - Compute the precision score

      * - :func:`~ComparisonReport.metrics.precision_recall`
        - Plot the precision-recall curve

      * - :func:`~ComparisonReport.metrics.prediction_error`
        - Plot the prediction error of a regression model

      * - :func:`~ComparisonReport.metrics.r2`
        - Compute the R² score

      * - :func:`~ComparisonReport.metrics.recall`
        - Compute the recall score

      * - :func:`~ComparisonReport.metrics.rmse`
        - Compute the root mean squared error

      * - :func:`~ComparisonReport.metrics.roc`
        - Plot the ROC curve

      * - :func:`~ComparisonReport.metrics.roc_auc`
        - Compute the ROC AUC score

      * - :func:`~ComparisonReport.metrics.summarize`
        - Report a set of metrics for the estimators

      * - :func:`~ComparisonReport.metrics.timings`
        - Get all measured processing times related to the different estimators


   **Inspection**: :class:`ComparisonReport.inspection`

   .. list-table::
      :widths: 30 70

      * - :func:`~ComparisonReport.inspection.coefficients`
        - Retrieve the coefficients for each report, including the intercepts




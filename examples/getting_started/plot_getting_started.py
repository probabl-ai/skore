"""
.. _example_getting_started:

======================
Skore: getting started
======================
"""

# %%
# This guide illustrates how to use skore through a complete
# machine learning workflow for binary classification:
#
# #. Set up a proper experiment with training and test data
# #. Develop and evaluate multiple models using cross-validation
# #. Compare models to select the best one
# #. Validate the final model on held-out data
# #. Track and organize your machine learning results
#
# Throughout this guide, we will see how skore helps you:
#
# * Avoid common pitfalls with smart diagnostics
# * Quickly get rich insights into model performance
# * Organize and track your experiments

# %%
# Setting up our binary classification problem
# ============================================
#
# Let's start by loading the German credit dataset, a classic binary classification
# problem where we predict the customer's credit risk ("good" or "bad").
#
# This dataset contains various features about credit applicants, including
# personal information, credit history, and loan details.

# %%
import pandas as pd
import skore
from sklearn.datasets import fetch_openml
from skrub import TableReport

german_credit = fetch_openml(data_id=31, as_frame=True, parser="pandas")
X, y = german_credit.data, german_credit.target
TableReport(german_credit.frame)

# %%
# Creating our experiment and held-out sets
# -----------------------------------------
#
# We will use skore's enhanced `train_test_split` function to create our experiment set
# and a left-out test set. The experiment set will be used for model development and
# cross-validation, while the left-out set will only be used at the end to validate
# our final model.
#
# Unlike scikit-learn's `train_test_split`, skore's version provides helpful diagnostics
# about potential issues with your data split, such as class imbalance.

# %%
X_experiment, X_holdout, y_experiment, y_holdout = skore.train_test_split(
    X, y, random_state=42
)

# %%
# skore tells us we have class-imbalance issues with our data, which we confirm
# with the `TableReport` above by clicking on the "class" column and looking at the
# class distribution: there are only 300 examples where the target is "bad".

# %%
# Model development with cross-validation
# =======================================
#
# We will investigate two different models using cross-validation:
#
# 1. A simple linear model with some preprocessing, powered by
#    :func:`skrub.tabular_pipeline`
# 2. A more advanced model which includes preprocessing, sklearn's
#    :class:`~sklearn.ensemble.HistGradientBoostingClassifier`
#
# Cross-validation is necessary to get a more reliable estimate of model performance.
# skore makes it easy through :class:`skore.CrossValidationReport`.

# %%
# Model no. 1: linear regression with preprocessing
# -------------------------------------------------
#
# Our first model will be a linear model.

# %%
from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline

simple_model = tabular_pipeline(LogisticRegression())
simple_model

# %%
# We now cross-validate the model with :class:`~skore.CrossValidationReport`.

# %%
from skore import CrossValidationReport

simple_cv_report = CrossValidationReport(
    simple_model,
    X=X_experiment,
    y=y_experiment,
    pos_label="good",
    splitter=5,
)

# %%
# The :meth:`~skore.CrossValidationReport.help` method shows all available methods and properties:

# %%
simple_cv_report.help()

# %%
# For example, we can examine the training data, which excludes the held-out data:

# %%
simple_cv_report.data.analyze()

# %%
# But we can also quickly get an overview of the performance of our model,
# using :meth:`~skore.CrossValidation.metrics.summarize`:

# %%

# `indicator_favorability=True` highlights whether higher or lower values are better
simple_metrics = simple_cv_report.metrics.summarize(indicator_favorability=True)
simple_metrics.frame()

# %%
# More complex metrics are available, such as the precision-recall curve:

# %%
precision_recall = simple_cv_report.metrics.precision_recall()
precision_recall

# %%
#
# .. note::
#
#     The output of :meth:`~skore.CrossValidation.precision_recall` is a
#     :class:`~skore.Display` object. This is a common pattern in skore which allows us
#     to access the information in several ways.

# %%
# As a plot to visualize the critical information:

# %%
precision_recall.plot()

# %%
# Or as a dataframe to access the raw information:

# %%
precision_recall.frame()

# %%
# Similarly, we can plot the confusion matrix:

# %%
confusion_matrix = simple_cv_report.metrics.confusion_matrix()
confusion_matrix.plot()

# %%
# Since our model is a linear model, we can study the importance that it gives
# to each feature:

# %%
coefficients = simple_cv_report.feature_importance.coefficients()
coefficients.frame()

# %%

# sphinx_gallery_start_ignore
# TODO: Replace with top_k argument when available
# sphinx_gallery_end_ignore

(
    coefficients.frame()
    .groupby("feature")
    .mean()
    .drop(columns=["split"])
    .sort_values(by="coefficients", key=abs, ascending=False)
    .head(15)[::-1]
    .plot.barh(
        title="Mean model weights",
        xlabel="Coefficient",
        ylabel="Feature",
    )
)

# %%
# Model no. 2: gradient boosting
# ------------------------------
#
# Now, we cross-validate a more advanced model using :class:`~sklearn.ensemble.HistGradientBoostingClassifier`.
# This model automatically handles preprocessing for different column types.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

advanced_model = HistGradientBoostingClassifier()
advanced_model

# %%
advanced_cv_report = CrossValidationReport(
    advanced_model, X=X_experiment, y=y_experiment, pos_label="good"
)

# %%
# We will now compare this new model with the previous one.

# %%
# Comparing our models
# ====================
#
# Now that we have our two models, we need to decide which one should go into production.
# We can compare them with a :class:`skore.ComparisonReport`.

# %%
from skore import ComparisonReport

comparison = ComparisonReport(
    {
        "Simple Linear Model": simple_cv_report,
        "Advanced Pipeline": advanced_cv_report,
    },
)

# %%
# This report also has a help menu:
comparison.help()

# %%
# In fact, it has mostly the same API as `CrossValidationReport` and we have access to
# the same tools to make statistical analysis and compare both models:
comparison_metrics = comparison.metrics.summarize(favorability=True)
comparison_metrics.frame()

# %%
comparison.metrics.precision_recall().plot()

# %%
# Based on the previous tables and plots, it seems that both models have similar
# performance. For the purposes of this guide, we make the arbitrary choice to deploy
# the linear model because it is more interpretable.

# %%
# Final model evaluation on held-out data
# =======================================
#
# Now that we have chosen to deploy the linear model, we will train it on
# the full experiment set and evaluate it on our held-out data: training on more data
# should help performance and we can also validate that our model generalizes well to
# new data.

# %%
from skore import EstimatorReport

final_report = EstimatorReport(
    simple_model,
    X_train=X_experiment,
    y_train=y_experiment,
    X_test=X_holdout,
    y_test=y_holdout,
    pos_label="good",
)

# %%
# :class:`skore.EstimatorReport` has a similar API to the other report classes:

# %%
final_metrics = final_report.metrics.summarize()
final_metrics.frame()

# %%
final_report.metrics.confusion_matrix().plot()

# %%
# We compare the performance on the held-out data with what we observed during cross-validation

# %%
pd.concat(
    [final_metrics.frame(), simple_cv_report.metrics.summarize().frame()],
    axis="columns",
)

# %%
# As expected, our final model gets better performance, likely thanks to the
# larger training set.

# %%
# Our final sanity check is to compare the features considered most impactful
# between our final model and the cross-validation

# %%
final_coefficients = final_report.feature_importance.coefficients()
final_top_15_features = (
    final_coefficients.frame()
    .sort_values(by="coefficients", key=abs, ascending=False)
    .head(15)["feature"]
    .reset_index(drop=True)
)

simple_coefficients = simple_cv_report.feature_importance.coefficients()
cv_top_15_features = (
    simple_coefficients.frame()
    .groupby("feature")
    .mean()
    .reset_index()
    .drop(columns=["split"])
    .sort_values(by="coefficients", key=abs, ascending=False)
    .head(15)["feature"]
    .reset_index(drop=True)
)

pd.concat(
    [final_top_15_features, cv_top_15_features], axis="columns", ignore_index=True
)

# %%
# They seem very similar, so we are done!

# %%
# Tracking our work with a skore Project
# ======================================
#
# Now that we have completed our modeling workflow, we should store our models in a
# safe place for future work; for example, if this research notebook were modified
# we would no longer be able to relate the current production model to the code that
# generated it.
#
# We can use a :class:`skore.Project` to keep track of our experiments.
# This makes it easy to organize, retrieve, and compare models over time.
#
# Usually this would be done as you go along the model development, but
# in the interest of simplicity we kept this until the end.

# %%

# sphinx_gallery_start_ignore
import os
import tempfile

temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["SKORE_WORKSPACE"] = temp_dir.name
# sphinx_gallery_end_ignore

# Load or create a local project
project = skore.Project("german_credit_classification")

# %%
# Store our reports with descriptive keys
project.put("simple_linear_model_cv", simple_cv_report)
project.put("advanced_pipeline_cv", advanced_cv_report)
project.put("final_model", final_report)

# %%
# Now we can retrieve a summary of our stored reports
from pprint import pprint

summary = project.summarize()
pprint(summary.reports())

# %%
# The :class:`~skore.project.summary.Summary` object provides an interactive widget in Jupyter notebooks
# that allows us to explore and filter your reports visually.
#
# Each line represents a model, and we can select models by clicking on lines
# or dragging on metric axes to filter by performance.
#
# In the following screenshot, we selected only the cross-validation reports;
# we will see that this allows us to retrieve exactly those reports.
#
# .. image:: /_static/images/screenshot_getting_started.png
#   :alt: Screenshot of the widget in a Jupyter notebook

# %%

# sphinx_gallery_start_ignore
# Pretend that the cross-validation reports were selected in the widget
summary = summary.query('report_type == "cross-validation"')
# sphinx_gallery_end_ignore

# Supposing you selected "Cross-validation" in the "Report type" tab,
# if you now call `reports()`, you get only the CrossValidationReports
pprint(summary.reports())

# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore

# %%
# .. admonition:: Stay tuned!
#
#   This is only the beginning for skore. We welcome your feedback and ideas
#   to make it the best tool for end-to-end data science.
#
#   Key benefits of using skore in your ML workflow:
#   * Standardized evaluation and comparison of models
#   * Rich visualizations and diagnostics
#   * Organized experiment tracking
#   * Seamless integration with scikit-learn
#
#   Feel free to join our community on `Discord <http://discord.probabl.ai>`_
#   or `create an issue <https://github.com/probabl-ai/skore/issues>`_.

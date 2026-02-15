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
# We will use skore's enhanced :func:`~skore.train_test_split` function to create our
# experiment set and a left-out test set. The experiment set will be used for model
# development and cross-validation, while the left-out set will only be used at the end
# to validate our final model.
#
# Unlike scikit-learn's :func:`~skore.train_test_split`, skore's version provides
# helpful diagnostics about potential issues with your data split, such as class
# imbalance.

# %%
X_experiment, X_holdout, y_experiment, y_holdout = skore.train_test_split(
    X, y, random_state=42
)

# %%
# Skore tells us we have class-imbalance issues with our data, which we confirm with the
# :class:`~skore.TableReport` above by clicking on the "class" column and looking at the
# class distribution: there are only 300 examples where the target is "bad". The second
# warning concerns time-ordered data, but our data does not contain time-ordered columns
# so we can safely ignore it.

# %%
# Model development with cross-validation
# =======================================
#
# We will investigate two different families of models using cross-validation.
#
# 1. A :class:`~sklearn.linear_model.LogisticRegression` which is a linear model
# 2. A :class:`~sklearn.ensemble.RandomForestClassifier` which is an ensemble of
#    decision trees.
#
# In both cases, we rely on :func:`skrub.tabular_pipeline` to choose the proper
# preprocessing depending on the kind of model.
#
# Cross-validation is necessary to get a more reliable estimate of model performance.
# skore makes it easy through :class:`skore.CrossValidationReport`.

# %%
# Model no. 1: Linear regression with preprocessing
# -------------------------------------------------
#
# Our first model will be a linear model, with automatic preprocessing of non-numeric
# data. Under the hood, skrub's :class:`~skrub.TableVectorizer` will adapt the
# preprocessing based on our choice to use a linear model.

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
# Skore reports allow us to structure the statistical information
# we look for when experimenting with predictive models. First, the
# :meth:`~skore.CrossValidationReport.help` method shows us all its available methods
# and attributes, with the knowledge that our model was trained for classification:

# %%
simple_cv_report.help()

# %%
# For example, we can examine the training data, which excludes the held-out data:

# %%
simple_cv_report.data.analyze()

# %%
# But we can also quickly get an overview of the performance of our model,
# using :meth:`~skore.CrossValidationReport.metrics.summarize`:

# %%
simple_metrics = simple_cv_report.metrics.summarize(favorability=True)
simple_metrics.frame()

# %%
# .. note::
#
#     `favorability=True` adds a column showing whether higher or lower metric values
#     are better.

# %%
# In addition to the summary of metrics, skore provides more advanced statistical
# information such as the precision-recall curve:

# %%
precision_recall = simple_cv_report.metrics.precision_recall()
precision_recall.help()

# %%
# .. note::
#
#     The output of :meth:`~skore.CrossValidation.precision_recall` is a
#     :class:`~skore.Display` object. This is a common pattern in skore which allows us
#     to access the information in several ways.

# %%
# We can visualize the critical information as a plot, with only a few lines of code:

# %%
precision_recall.plot()

# %%
# Or we can access the raw information as a dataframe if additional analysis is needed:

# %%
precision_recall.frame()

# %%
# As another example, we can plot the confusion matrix with the same consistent API:

# %%
confusion_matrix = simple_cv_report.metrics.confusion_matrix()
confusion_matrix.plot()

# %%
# Skore also provides utilities to inspect models. Since our model is a linear
# model, we can study the importance that it gives to each feature:

# %%
coefficients = simple_cv_report.inspection.coefficients()
coefficients.frame()

# %%
coefficients.plot(select_k=15)

# %%
# Model no. 2: Random forest
# --------------------------
#
# Now, we cross-validate a more advanced model using
# :class:`~sklearn.ensemble.RandomForestClassifier`. Again, we rely on
# :func:`~skrub.tabular_pipeline` to perform the appropriate preprocessing to use with
# this model.

# %%
from sklearn.ensemble import RandomForestClassifier

advanced_model = tabular_pipeline(RandomForestClassifier(random_state=0))
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
# Now that we have our two models, we need to decide which one should go into
# production. We can compare them with a :class:`skore.ComparisonReport`.

# %%
from skore import ComparisonReport

comparison = ComparisonReport(
    {
        "Simple Linear Model": simple_cv_report,
        "Advanced Pipeline": advanced_cv_report,
    },
)

# %%
# This report follows the same API as :class:`~skore.CrossValidationReport`:
comparison.help()

# %%
# We have access to the same tools to perform statistical analysis and compare both
# models:
comparison_metrics = comparison.metrics.summarize(favorability=True)
comparison_metrics.frame()

# %%
comparison.metrics.precision_recall().plot()

# %%
# Based on the previous tables and plots, it seems that the
# :class:`~sklearn.ensemble.RandomForestClassifier` model has slightly better
# performance. For the purposes of this guide however, we make the arbitrary choice
# to deploy the linear model to make a comparison with the coefficients study shown
# earlier.

# %%
# Final model evaluation on held-out data
# =======================================
#
# Now that we have chosen to deploy the linear model, we will train it on the full
# experiment set and evaluate it on our held-out data: training on more data should help
# performance and we can also validate that our model generalizes well to new data. This
# can be done in one step with :meth:`~skore.ComparisonReport.create_estimator_report`.

# %%

final_report = comparison.create_estimator_report(
    name="Simple Linear Model", X_test=X_holdout, y_test=y_holdout
)

# %%
# This returns a :class:`~skore.EstimatorReport` which has a similar API to the other
# report classes:

# %%
final_metrics = final_report.metrics.summarize()
final_metrics.frame()

# %%
final_report.metrics.confusion_matrix().plot()

# %%
# We can easily combine the results of the previous cross-validation together with
# the evaluation on the held-out dataset, since the two are accessible as dataframes.
# This way, we can check if our chosen model meets the expectations we set during the
# experiment phase.

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
# between our final model and the cross-validation:

# %%

# sphinx_gallery_start_ignore
# TODO: Use native aggregation when available
# sphinx_gallery_end_ignore

final_coefficients = final_report.inspection.coefficients()
final_top_15_features = final_coefficients.frame(select_k=15)["feature"]

simple_coefficients = simple_cv_report.inspection.coefficients()
cv_top_15_features = (
    simple_coefficients.frame(select_k=15)
    .groupby("feature", sort=False)
    .mean()
    .drop(columns="split")
    .reset_index()["feature"]
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
# safe place for future work. Indeed, if this research notebook were modified,
# we would no longer be able to relate the current production model to the code that
# generated it.
#
# We can use a :class:`skore.Project` to keep track of our experiments.
# This makes it easy to organize, retrieve, and compare models over time.
#
# Usually this would be done as you go along the model development, but
# in the interest of simplicity we kept this until the end.

# %%
# We load or create a local project:

# %%

# sphinx_gallery_start_ignore
import os
import tempfile

temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["SKORE_WORKSPACE"] = temp_dir.name
# sphinx_gallery_end_ignore

project = skore.Project("german_credit_classification")

# %%
# We store our reports with descriptive keys:

# %%
project.put("simple_linear_model_cv", simple_cv_report)
project.put("advanced_pipeline_cv", advanced_cv_report)
project.put("final_model", final_report)

# %%
# Now we can retrieve a summary of our stored reports:

# %%
summary = project.summarize()
# Uncomment the next line to display the widget in an interactive environment:
# summary

# %%
# .. note::
#     Calling `summary` in a Jupyter notebook cell will show the following parallel
#     coordinate plot to help you select models that you want to retrieve:
#
#     .. image:: /_static/images/screenshot_getting_started.png
#       :alt: Screenshot of the widget in a Jupyter notebook
#
#     Each line represents a model, and we can select models by clicking on lines
#     or dragging on metric axes to filter by performance.
#
#     In the screenshot, we selected only the cross-validation reports;
#     this allows us to retrieve exactly those reports programmatically.

# %%
# Supposing you selected "Cross-validation" in the "Report type" tab, if you now call
# :meth:`~skore.project._summary.Summary.reports`, you get only the
# :class:`~skore.CrossValidationReport` objects, which
# you can directly put in the form of a :class:`~skore.ComparisonReport`:

# %%

# sphinx_gallery_start_ignore
# Pretend that the cross-validation reports were selected in the widget
summary = summary.query('report_type == "cross-validation"')
# sphinx_gallery_end_ignore

new_report = summary.reports(return_as="comparison")
new_report.help()

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
#
#   * Standardized evaluation and comparison of models
#   * Rich visualizations and diagnostics
#   * Organized experiment tracking
#   * Seamless integration with scikit-learn
#
#   Feel free to join our community on `Discord <https://discord.probabl.ai>`_
#   or `create an issue <https://github.com/probabl-ai/skore/issues>`_.

"""
.. _example_use_case_employee_salaries:

==============================================
Simplified and structured experiment reporting
==============================================

This example shows how to leverage `skore` for structuring useful experiment information
allowing to get insights from machine learning experiments.
"""

# %%
#
# We set some environment variables to avoid some spurious warnings related to
# parallelism when using the `TextEncoder` from `skrub` that uses tokenizers.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# %%
# Loading a non-trivial dataset
# =============================

# %%
#
# We use a skrub dataset that is non-trivial that contains information about employees
# and their salaries.
from skrub.datasets import fetch_employee_salaries

datasets = fetch_employee_salaries()
df, y = datasets.X, datasets.y

# %%
#
# Let's first have a condensed summary of the input data using a
# :class:`skrub.TableReport`.
from skrub import TableReport

table_report = TableReport(df)
table_report

# %%
# From the table report, we can make a few observations:
#
# * The type of data is heterogeneous: we mainly have categorical and date-related
#   features.
#
# * The year related to the ``date_first_hired`` column is also present in the
#   ``date`` column.
#   Hence, we should beware of not creating twice the same feature during the feature
#   engineering.
#
# * By looking at the "Associations" tab of the table report, we observe that two
#   features are holding the exact same information: ``department`` and
#   ``department_name``.
#   Hence, during our feature engineering, we could potentially drop one of them if the
#   final predictive model is sensitive to the collinearity.
#
# * When looking at the "Stats" tab, we observe that the ``division`` and
#   ``employee_position_title`` are two features containing a large number of
#   categories.
#   It is something that we should consider in our feature engineering.
#
# In terms of target and thus the task that we want to solve, we are interested in
# predicting the salary of an employee given the previous features. We therefore have
# a regression task at end.
y

# %%
#
# Later, we will show that `skore` stores similar information when a model is trained on
# some dataset to allow to get insights on the dataset used to train and test the model.

# %%
# Tree-based model
# ================
#
# Let's start by creating a tree-based model using some out-of-the-box tools.
#
# For feature engineering we use skrub's :class:`~skrub.TableVectorizer`.
# To deal with the high cardinality of the categorical features, we use a
# :class:`~skrub.TextEncoder` that uses a language model and an embedding model to
# encode the categorical features.
#
# Finally, we use a :class:`~sklearn.ensemble.HistGradientBoostingRegressor` as a
# base estimator that is a rather robust model.
#
# Modelling
# ^^^^^^^^^

from skrub import TableVectorizer, TextEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TableVectorizer(high_cardinality=TextEncoder()),
    HistGradientBoostingRegressor(),
)
model

# %%
# Evaluation
# ^^^^^^^^^^
#
# Let us compute the cross-validation report for this model using
# :class:`skore.CrossValidationReport`:
from skore import CrossValidationReport

hgbt_model_report = CrossValidationReport(
    estimator=model, X=df, y=y, cv_splitter=5, n_jobs=4
)
hgbt_model_report.help()

# %%
#
# A report provides a collection of useful information. For instance, it allows to
# compute on demand the predictions of the model and some performance metrics.
#
# Let's cache once for all the predictions of the cross-validated models.
hgbt_model_report.cache_predictions(n_jobs=4)

# %%
# Now, that the predictions are cached, any request to compute a metric will be
# performed using those cached predictions and will be fast.
#
# Let's have a look at the some standard performance metrics that are chosen based on
# the task at hand which is a regression task.
hgbt_model_report.metrics.summarize(indicator_favorability=True).frame()

# %%
#
# We get the results from some statistical metrics aggregated over the cross-validation
# splits as well as some performance metrics related to the time it took to train and
# test the model. The favorability of each metric indicated whether the metric is better
# when higher or lower.
#
# The :class:`skore.CrossValidationReport` also provides a way to inspect similar
# information at the level of each cross-validation split by accessing an
# :class:`skore.EstimatorReport` for each split.
hgbt_split_1 = hgbt_model_report.estimator_reports_[0]
hgbt_split_1.metrics.summarize(indicator_favorability=True).frame()

# %%
#
# We also have access to some additional information regarding the dataset used for
# training and testing of the model similar to what we have seen in the previous
# section. For instance, let's check some information about the training dataset.
train_data_display = hgbt_split_1.data.analyze(data_source="train")
train_data_display

# %%
#
# The display obtained allows to get a quick overview with the same HTML-based view
# than the :class:`skore.TableReport` we have seen earlier. In addition, you can access
# a :meth:`skore.TableReportDisplay.plot` method to have a particular focus on one
# potential analysis. For instance, we can get a figure representing the correlation
# matrix of the training dataset.
train_data_display.plot(kind="corr")

# %%
#
# This display also provides more information to plot 1-d or 2-d distributions of some
# columns of the dataset.
train_data_display.plot(y="gender")

# %%
train_data_display.plot(y="gender", hue="current_annual_salary")

# %%
# Linear model
# ============
#
# Now that we have established a first model that serves as a baseline,
# we shall proceed to define a quite complex linear model
# (a pipeline with a complex feature engineering that uses
# a linear model as the base estimator).

# %%
# Modelling
# ^^^^^^^^^

import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.linear_model import RidgeCV
from skrub import DatetimeEncoder, ToDatetime, DropCols, GapEncoder


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


one_hot_features = ["gender", "department_name", "assignment_category"]
datetime_features = "date_first_hired"

date_encoder = make_pipeline(
    ToDatetime(),
    DatetimeEncoder(resolution="day", add_weekday=True, add_total_seconds=False),
    DropCols("date_first_hired_year"),
)

date_engineering = make_column_transformer(
    (periodic_spline_transformer(12, n_splines=6), ["date_first_hired_month"]),
    (periodic_spline_transformer(31, n_splines=15), ["date_first_hired_day"]),
    (periodic_spline_transformer(7, n_splines=3), ["date_first_hired_weekday"]),
)

feature_engineering_date = make_pipeline(date_encoder, date_engineering)

preprocessing = make_column_transformer(
    (feature_engineering_date, datetime_features),
    (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), one_hot_features),
    (GapEncoder(n_components=100), "division"),
    (GapEncoder(n_components=100), "employee_position_title"),
)

model = make_pipeline(preprocessing, RidgeCV(alphas=np.logspace(-3, 3, 100)))
model

# %%
# In the diagram above, we can see what how we performed our feature engineering:
#
# * For categorical features, we use two approaches: if the number of categories is
#   relatively small, we use a `OneHotEncoder` and if the number of categories is
#   large, we use a `GapEncoder` that was designed to deal with high cardinality
#   categorical features.
#
# * Then, we have another transformation to encode the date features. We first split the
#   date into multiple features (day, month, and year). Then, we apply a periodic spline
#   transformation to each of the date features to capture the periodicity of the data.
#
# * Finally, we fit a :class:`~sklearn.linear_model.RidgeCV` model.

# %%
# Evaluation
# ^^^^^^^^^^
#
# Now, we want to evaluate this linear model via cross-validation (with 5 folds).
# For that, we use skore's :class:`~skore.CrossValidationReport` to investigate the
# performance of our model.
linear_model_report = CrossValidationReport(
    estimator=model, X=df, y=y, cv_splitter=5, n_jobs=4
)
linear_model_report.help()

# %%
# We observe that the cross-validation report detected that we have a regression task
# and provides us with some metrics and plots that make sense for our
# specific problem at hand.
#
# To accelerate any future computation (e.g. of a metric), we cache once and for all the
# predictions of our model.
# Note that we do not necessarily need to cache the predictions as the report will
# compute them on the fly (if not cached) and cache them for us.

# %%
import warnings

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    linear_model_report.cache_predictions(n_jobs=4)

# %%
# We can now have a look at the performance of the model with some standard metrics.
linear_model_report.metrics.summarize(indicator_favorability=True).frame()

# %%
# Comparing the models
# ====================
#
# Now that we cross-validated our models, we can make some further comparison using the
# :class:`skore.ComparisonReport`:

# %%
from skore import ComparisonReport

comparator = ComparisonReport([hgbt_model_report, linear_model_report])
comparator.metrics.summarize(indicator_favorability=True).frame()

# %%
# In addition, if we forgot to compute a specific metric
# (e.g. :func:`~sklearn.metrics.mean_absolute_error`),
# we can easily add it to the report, without re-training the model and even
# without re-computing the predictions since they are cached internally in the report.
# This allows us to save some potentially huge computation time.

# %%
from sklearn.metrics import get_scorer

scoring = ["r2", "rmse", get_scorer("neg_mean_absolute_error")]
scoring_kwargs = {"response_method": "predict"}
scoring_names = ["R²", "RMSE", "MAE"]

comparator.metrics.summarize(
    scoring=scoring,
    scoring_kwargs=scoring_kwargs,
    scoring_names=scoring_names,
    indicator_favorability=True,
).frame()

# %%
# Finally, we can even get a deeper understanding by analyzing each fold in the
# :class:`~skore.CrossValidationReport`.
# Here, we plot the actual-vs-predicted values for each fold.
import matplotlib.pyplot as plt

linear_model_report.metrics.prediction_error().plot(kind="actual_vs_predicted")

plt.tight_layout()

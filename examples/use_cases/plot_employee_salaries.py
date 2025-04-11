"""
.. _example_use_case_employee_salaries:

===============================
Simplified experiment reporting
===============================

This example shows how to leverage skore for reporting model evaluation and
storing the results for further analysis.
"""

# %%
#
# We set some environment variables to avoid some spurious warnings related to
# parallelism.
import os

os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# %%
# Creating a skore project and loading some data
# ==============================================

# %%
#
# Let's open a skore project in which we will be able to store artifacts from our
# experiments.
import skore

# sphinx_gallery_start_ignore
import os
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = Path(temp_dir.name)
os.chdir(temp_dir_path)
# sphinx_gallery_end_ignore
my_project = skore.Project("my_project")

# %%
#
# We use a skrub dataset that is non-trivial.
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
# We can store the report in the skore project so that we can easily retrieve it later
# without necessarily having to reload the dataset and recompute the report.
my_project.put("Input data summary", table_report)

# %%
#
# In terms of target and thus the task that we want to solve, we are interested in
# predicting the salary of an employee given the previous features. We therefore have
# a regression task at end.
y

# %%
# Tree-based model
# ===============

# %%
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
# Let us compute the cross-validation report for this model using :class:`skore.CrossValidationReport`:
from skore import CrossValidationReport

report = CrossValidationReport(estimator=model, X=df, y=y, cv_splitter=5, n_jobs=4)
report.help()

# %%
#
# We cache the predictions for later use.
report.cache_predictions(n_jobs=4)

# %%
#
# We store the report in our skore project.
my_project.put("HGBT model report", report)

# %%
#
# We can now have a look at the performance of the model with some standard metrics.
report.metrics.report_metrics()


# %%
# Linear model
# =========
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
report = CrossValidationReport(estimator=model, X=df, y=y, cv_splitter=5, n_jobs=4)
report.help()

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
    report.cache_predictions(n_jobs=4)

# %%
# To ensure this cross-validation report is not lost, let us save it in our skore
# project.
my_project.put("Linear model report", report)

# %%
# We can now have a look at the performance of the model with some standard metrics.
report.metrics.report_metrics(indicator_favorability=True)

# %%
# Comparing the models
# ====================
#
# At this point, we may not have been cautious and could have already overwritten the
# report and model from our initial (tree-based model) attempt.
# Fortunately, since we saved the reports in our skore project, we can easily recover
# them.
# So, let us retrieve those reports.

hgbt_model_report = my_project.get("HGBT model report")
linear_model_report = my_project.get("Linear model report")

# %%
#
# Now that we retrieved the reports, we can make some further comparison and build upon
# some usual pandas operations to concatenate the results.
import pandas as pd

results = pd.concat(
    [
        hgbt_model_report.metrics.report_metrics(),
        linear_model_report.metrics.report_metrics(),
    ],
    axis=1,
)
results

# %%
#
# In addition, if we forgot to compute a specific metric
# (e.g. :func:`~sklearn.metrics.mean_absolute_error`),
# we can easily add it to the report, without re-training the model and even
# without re-computing the predictions since they are cached internally in the report.
# This allows us to save some potentially huge computation time.
from sklearn.metrics import mean_absolute_error

scoring = ["r2", "rmse", mean_absolute_error]
scoring_kwargs = {"response_method": "predict"}
scoring_names = ["R2", "RMSE", "MAE"]
results = pd.concat(
    [
        hgbt_model_report.metrics.report_metrics(
            scoring=scoring,
            scoring_kwargs=scoring_kwargs,
            scoring_names=scoring_names,
        ),
        linear_model_report.metrics.report_metrics(
            scoring=scoring,
            scoring_kwargs=scoring_kwargs,
            scoring_names=scoring_names,
        ),
    ],
    axis=1,
)
results

# %%
# .. note::
#   We could have also used the :class:`skore.ComparisonReport` to compare estimator
#   reports.
#   This is done in :ref:`example_feature_importance`.

# %%
#
# Finally, we can even get the individual :class:`~skore.EstimatorReport` for each fold
# from the cross-validation to make further analysis.
# Here, we plot the actual vs predicted values for each fold.
from itertools import zip_longest
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 18))
for split_idx, (ax, estimator_report) in enumerate(
    zip_longest(axs.flatten(), linear_model_report.estimator_reports_)
):
    if estimator_report is None:
        ax.axis("off")
        continue
    estimator_report.metrics.prediction_error().plot(kind="actual_vs_predicted", ax=ax)
    ax.set_title(f"Split #{split_idx + 1}")
    ax.legend(loc="lower right")

plt.tight_layout()
# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore

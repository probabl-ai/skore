"""
.. _example_use_case_employee_salaries:

===============================
Simplified experiment reporting
===============================

This example shows how to leverage `skore` for reporting model evaluation and
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
#
# Let's open a `skore` project in which we will be able to store artifacts from our
# experiments.
import skore

project = skore.open("my_project", create=True)

# %%
#
# We use a `skrub` dataset that is non-trivial dataset.
from skrub.datasets import fetch_employee_salaries

datasets = fetch_employee_salaries()
df, y = datasets.X, datasets.y

# %%
#
# Let's first have a condensed summary of the input data using
# :class:`~skrub.TableReport`.
from skrub import TableReport

table_report = TableReport(df)
table_report

# %%
#
# First, we can check that the type of data is heterogeneous: we mainly have categorical
# features and feature related to dates.
#
# We can observe that the year related to the first hired is also present in the date.
# Hence, we should beware of not creating twice the same feature during the feature
# engineering.
#
# By looking at the "Associations" tab, we observe that two features are exactly holding
# the same information: "department" and "department_name". So during our feature
# engineering, we could potentially drop one of them if the final predictive model
# is sensitive to the collinearity.
#
# When looking at the "Stats" tab, we observe that the "division" and
# "employee_position_title" are two features containing a large number of categories. It
# something that we should consider in our feature engineering.
#
# We can store the report in the project so that we can easily retrieve it later
# without necessarily having to reload the dataset and recomputing the report.
project.put("Input data summary", table_report)

# %%
#
# In terms of target and thus the task that we want to solve, we are interested in
# predicting the salary of an employee given the previous features. We therefore have
# a regression task at end.
y

# %%
#
# In a first attempt, we will define a rather complex predictive model that will use
# a linear model as a base estimator.
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
#
# In the diagram above, we can see what we intend to do as feature engineering.
# For categorical features, we use two approaches: if the number of categories is
# relatively small, we use a `OneHotEncoder` and if the number of categories is
# large, we use a `GapEncoder` that was designed to deal with high cardinality
# categorical features.
#
# Then, we have another transformation to encode the date features. We first split the
# date into multiple features (day, month, and year). Then, we apply a periodic spline
# transformation to each of the date features to capture the periodicity of the data.
#
# Finally, we fit a :class:`~sklearn.linear_model.RidgeCV` model.
#
# Now, we want to evaluate this complex model via cross-validation. We would like to
# use 5 folds. We use :class:`~skore.CrossValidationReport` to allow us to investigate
# the performance of the model.
from skore import CrossValidationReport

report = CrossValidationReport(estimator=model, X=df, y=y, cv_splitter=5, n_jobs=3)
report.help()

# %%
#
# We observe that the report detected that we have a regression task and provide us only
# a subset of the metrics and plots that make sense for our problem at hand. To later
# accelerate the computation, we cache once for all the predictions of the model. Note
# that we don't necessarily need to cache the predictions as the report will compute
# them on the fly if not cached and cache them for us.

# %%
import warnings

with warnings.catch_warnings():
    # catch the warnings raised by the OneHotEncoder for seeing unknown categories
    # at transform time
    warnings.simplefilter(action="ignore", category=UserWarning)
    report.cache_predictions(n_jobs=3)

# %%
#
# To not lose the report, let's store it in our `skore` project.
project.put("Linear model report", report)

# %%
#
# We can now have a look at the performance of the model with some standard metrics.
report.metrics.report_metrics(aggregate=["mean", "std"])

# %%
#
# So now, that we have our first baseline model, we can try an out-of-the-box model
# using `skrub` that makes feature engineering for us. To deal with the high
# cardinality of the categorical features, we use a :class:`~skrub.TextEncoder` that
# use a language model to embed the categorical features.
#
# Finally, we use a :class:`~sklearn.ensemble.HistGradientBoostingRegressor` as a
# base estimator that is a rather robust model.
from skrub import TableVectorizer, TextEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TableVectorizer(high_cardinality=TextEncoder()),
    HistGradientBoostingRegressor(),
)
model

# %%
#
# Let's compute the cross-validation report for this model.
from skore import CrossValidationReport

report = CrossValidationReport(estimator=model, X=df, y=y, cv_splitter=5, n_jobs=3)
report.help()

# %%
#
# We cache the predictions for later use.
report.cache_predictions(n_jobs=3)

# %%
#
# We store the report in our `skore` project.
project.put("HGBDT model report", report)

# %%
#
# We can now have a look at the performance of the model with some standard metrics.
report.metrics.report_metrics(aggregate=["mean", "std"])

# %%
#
# At this stage, I might not been careful and have already overwritten the report and
# model from my first attempt. Hopefully, because we stored the reports in our `skore`
# project, we can easily retrieve them. So let's retrieve the reports.
linear_model_report = project.get("Linear model report")
hgbdt_model_report = project.get("HGBDT model report")

# %%
#
# Now that we retrieved the reports, I can make further comparison and build upon some
# usual pandas operations to concatenate the results.
import pandas as pd

results = pd.concat(
    [
        linear_model_report.metrics.report_metrics(aggregate=["mean", "std"]),
        hgbdt_model_report.metrics.report_metrics(aggregate=["mean", "std"]),
    ]
)
results

# %%
#
# In addition, if I forget to compute a specific metric, I can easily add it to the
# the report, without retraining the model and even recomputing the predictions since
# they are cached internally in the report. It allows to save some time.
from sklearn.metrics import mean_absolute_error

scoring = ["r2", "rmse", mean_absolute_error]
scoring_kwargs = {"response_method": "predict"}
scoring_names = ["R2", "RMSE", "MAE"]
results = pd.concat(
    [
        linear_model_report.metrics.report_metrics(
            scoring=scoring,
            scoring_kwargs=scoring_kwargs,
            scoring_names=scoring_names,
            aggregate=["mean", "std"],
        ),
        hgbdt_model_report.metrics.report_metrics(
            scoring=scoring,
            scoring_kwargs=scoring_kwargs,
            scoring_names=scoring_names,
            aggregate=["mean", "std"],
        ),
    ]
)
results

# %%
#
# Finally, we can even get individual :class:`~skore.EstimatorReport` from the
# cross-validation to make further analysis. Here, we plot the actual vs predicted
# values for each of the splits.
from itertools import zip_longest
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 18))
for split_idx, (ax, estimator_report) in enumerate(
    zip_longest(axs.flatten(), linear_model_report.estimator_reports_)
):
    if estimator_report is None:
        ax.axis("off")
        continue
    estimator_report.metrics.plot.prediction_error(kind="actual_vs_predicted", ax=ax)
    ax.set_title(f"Split #{split_idx + 1}")
    ax.legend(loc="lower right")
plt.tight_layout()

# %%
#
# Finally, we clean up the project by removing the temporary directory.
project.clear()

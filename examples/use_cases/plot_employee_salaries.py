# %%
import os
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# %%
import skore

project = skore.open(temp_dir_path / "my_project")

# %%
from skrub.datasets import fetch_employee_salaries

datasets = fetch_employee_salaries()
df, y = datasets.X, datasets.y

# %%
from skrub import TableReport

table_report = TableReport(df)
table_report

# %%
project.put("df_report", table_report)

# %%
y

# %%
import numpy as np
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from skrub import DatetimeEncoder, ToDatetime, DropCols
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV


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


categorical_features = [
    "gender",
    "department_name",
    "division",
    "assignment_category",
    "employee_position_title",
    "year_first_hired",
]
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
    (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_features),
)

# %%
linear_model = make_pipeline(preprocessing, RidgeCV(alphas=np.logspace(-3, 3, 100)))
linear_model

# %%
from skore import CrossValidationReport

linear_model_report = CrossValidationReport(
    estimator=linear_model, X=df, y=y, cv_splitter=10, n_jobs=3
)
linear_model_report.help()

# %%
linear_model_report.cache_predictions(n_jobs=3)

# %%
project.put("linear_model_report", linear_model_report)

# %%
linear_model_report.metrics.report_metrics(aggregate=["mean", "std"])

# %%
from skrub import TableVectorizer, TextEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

hgbdt_model = make_pipeline(
    TableVectorizer(high_cardinality=TextEncoder()),
    HistGradientBoostingRegressor(),
)

# %%
from skore import CrossValidationReport

hgbdt_model_report = CrossValidationReport(
    estimator=hgbdt_model, X=df, y=y, cv_splitter=10, n_jobs=3
)
hgbdt_model_report.help()

# %%
hgbdt_model_report.cache_predictions(n_jobs=3)

# %%
project.put("hgbdt_model_report", hgbdt_model_report)

# %%
hgbdt_model_report.metrics.report_metrics(aggregate=["mean", "std"])

# %%
import pandas as pd

results = pd.concat(
    [
        linear_model_report.metrics.report_metrics(aggregate=["mean", "std"]),
        hgbdt_model_report.metrics.report_metrics(aggregate=["mean", "std"]),
    ]
)
results

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=2, nrows=5, figsize=(12, 25))
for split_idx, (ax, estimator_report) in enumerate(
    zip(axs.flatten(), linear_model_report.estimator_reports_)
):
    estimator_report.metrics.plot.prediction_error(kind="actual_vs_predicted", ax=ax)
    ax.set_title(f"Split #{split_idx + 1}")
    ax.legend(loc="lower right")
plt.tight_layout()

# %%
reloading_report = project.get("linear_model_report")
reloading_report.help()

# %%
reloading_report.metrics.report_metrics(aggregate=["mean", "std"])

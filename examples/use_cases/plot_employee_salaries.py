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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

feature_engineering_date = Pipeline(
    steps=[
        (
            "date_encoding",
            Pipeline(
                steps=[
                    ("to_datetime", ToDatetime()),
                    (
                        "datetime_encoder",
                        DatetimeEncoder(
                            resolution="day", add_weekday=True, add_total_seconds=False
                        ),
                    ),
                    ("drop_year", DropCols("date_first_hired_year")),
                ]
            ),
        ),
        (
            "date_engineering",
            ColumnTransformer(
                transformers=[
                    (
                        "date_month",
                        periodic_spline_transformer(12, n_splines=6),
                        ["date_first_hired_month"],
                    ),
                    (
                        "date_day",
                        periodic_spline_transformer(31, n_splines=15),
                        ["date_first_hired_day"],
                    ),
                    (
                        "date_weekday",
                        periodic_spline_transformer(7, n_splines=3),
                        ["date_first_hired_weekday"],
                    ),
                ]
            ),
        ),
    ]
)
preprocessing = ColumnTransformer(
    transformers=[
        ("date_engineering", feature_engineering_date, datetime_features),
        (
            "categorical",
            OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)
linear_model = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("linear_model", RidgeCV(alphas=np.logspace(-3, 3, 100))),
    ]
)
linear_model

# %%
from skore import CrossValidationReport

linear_model_report = CrossValidationReport(
    estimator=linear_model, X=df, y=y, cv_splitter=10, n_jobs=3
)
linear_model_report.help()

# %%
for report in linear_model_report.estimator_reports_:
    report._parent_progress = None
    report._progress_info = None

# %%
project.put("linear_model_report", linear_model_report.estimator_reports_[0])

# %%
linear_model_report.cache_predictions(n_jobs=3)

# %%
linear_model_report.metrics.report_metrics(aggregate=["mean", "std"])

# %%
from skrub import TableVectorizer, TextEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TableVectorizer(high_cardinality=TextEncoder()),
    HistGradientBoostingRegressor(),
)

# %%
from skore import CrossValidationReport

report = CrossValidationReport(model, X=df, y=y, cv_splitter=10, n_jobs=3)
report.help()

# %%
report.metrics.report_metrics()

# %%
report.metrics.report_metrics(aggregate=["mean", "std"])

# %%

"""
.. _example_pandas styling:

==========================================
Applying some pandas styling to dataframes
==========================================
"""

# %%
# From the source code
# ====================

# %%
from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor

metrics_dict = _MetricsAccessor._SCORE_OR_LOSS_INFO

# %%
# Regression task
# ===============

# %%
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
X, y = california_housing.data, california_housing.target
california_housing.frame.head(2)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

estimators = [
    make_pipeline(StandardScaler(), LinearRegression()),
    make_pipeline(StandardScaler(), Ridge(random_state=0)),
    RandomForestRegressor(random_state=0),
    HistGradientBoostingRegressor(random_state=0),
]

# %%
from skore import EstimatorReport

estimator_reports = [
    EstimatorReport(est, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    for est in estimators
]

# %%
from skore import ComparisonReport

comparator = ComparisonReport(reports=estimator_reports)

# %%
df = comparator.metrics.report_metrics()
df

# %%
import pandas as pd

# Automatically determine lower_is_better and higher_is_better
lower_is_better = [
    item["name"] for item in metrics_dict.values() if item["icon"] == "(↘︎)"
]
higher_is_better = [
    item["name"] for item in metrics_dict.values() if item["icon"] == "(↗︎)"
]


# Function to determine the best value and apply bold styling
def highlight_best(row):
    metric = row.name
    if metric in lower_is_better:
        best_value = row.min()  # Lower is better
        is_best = row == best_value
    else:  # higher_is_better
        best_value = row.max()  # Higher is better
        is_best = row == best_value
    styles = ["font-weight: bold" if x else "" for x in is_best]
    return styles


# Function to apply a blue gradient based on whether lower or higher is better
def apply_gradient(row):
    metric = row.name
    if metric in lower_is_better:
        # For lower-is-better, smaller values get a stronger blue
        gradient = pd.Series(
            [
                "background-color: rgba(0, 0, 255, {:.2f})".format(
                    1 - (x - row.min()) / (row.max() - row.min())
                )
                for x in row
            ],
            index=row.index,
        )
    else:
        # For higher-is-better, larger values get a stronger blue
        gradient = pd.Series(
            [
                "background-color: rgba(0, 0, 255, {:.2f})".format(
                    (x - row.min()) / (row.max() - row.min())
                )
                for x in row
            ],
            index=row.index,
        )
    return gradient


# Apply the styling
styled_df = (
    df.style.apply(highlight_best, axis=1)  # Bold the best value
    .apply(apply_gradient, axis=1)  # Apply blue gradient
    .format("{:.6f}")
)  # Format numbers to 6 decimal places

# Display the styled DataFrame (in Jupyter Notebook)
styled_df

# %%
# Classification task
# ===================
#
# .. warning::
#
#   This part is not done yet.

# %%
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.ensemble import RandomForestClassifier
from skore import EstimatorReport

rf = RandomForestClassifier(random_state=0)
rf_report = EstimatorReport(
    rf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
gb_report = EstimatorReport(
    gb, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
comparator = ComparisonReport(reports=[rf_report, gb_report])

# %%
df = comparator.metrics.report_metrics(pos_label=1, indicator_favorability=False)
df

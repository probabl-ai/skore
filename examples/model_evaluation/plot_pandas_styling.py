"""
.. _example_pandas styling:

==========================================
Applying some pandas styling to dataframes
==========================================
"""

# %%
# TODO
# ====
#
# - include ``indicator_favorability=True``

# %%
# From the source code
# ====================

# %%
from skore.sklearn._estimator.metrics_accessor import _MetricsAccessor
from pprint import pprint

metrics_dict = _MetricsAccessor._SCORE_OR_LOSS_INFO
pprint(metrics_dict)

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
df = comparator.metrics.report_metrics(indicator_favorability=False)
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


def highlight_best(row):
    """Determine the best value and apply bold styling."""
    metric = row.name
    if metric in lower_is_better:
        best_value = row.min()  # Lower is better
        is_best = row == best_value
    else:  # higher_is_better
        best_value = row.max()  # Higher is better
        is_best = row == best_value
    styles = ["font-weight: bold" if x else "" for x in is_best]
    return styles


def apply_gradient(row):
    """Apply a color gradient based on whether lower or higher is better."""
    metric = row.name
    if metric in lower_is_better:
        # For lower-is-better, smaller values get a stronger color
        gradient = pd.Series(
            [
                "background-color: rgba(30, 34, 170, {:.2f})".format(
                    1 - (x - row.min()) / (row.max() - row.min())
                )
                for x in row
            ],
            index=row.index,
        )
    else:
        # For higher-is-better, larger values get a stronger color
        gradient = pd.Series(
            [
                "background-color: rgba(30, 34, 170, {:.2f})".format(
                    (x - row.min()) / (row.max() - row.min())
                )
                for x in row
            ],
            index=row.index,
        )
    return gradient


styled_df = (
    df.style.apply(highlight_best, axis=1)
    .apply(apply_gradient, axis=1)
    .format("{:.3f}")
)

styled_df

# %%
# Classification task
# ===================

# %%
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

estimators = [
    make_pipeline(StandardScaler(), LogisticRegression()),
    RandomForestClassifier(random_state=0),
    HistGradientBoostingClassifier(random_state=0),
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
df = comparator.metrics.report_metrics(pos_label=1, indicator_favorability=False)
df

# %%
styled_df = (
    df.style.apply(highlight_best, axis=1)
    .apply(apply_gradient, axis=1)
    .format("{:.3f}")
)

styled_df

"""
.. _example_pandas styling:

==========================================
Applying some pandas styling to dataframes
==========================================

This example shows how to apply styling to the metrics dataframe returned by
:meth:`skore.ComparisonReport.metrics.report_metrics`. The styling helps highlight
which estimator performs better for each metric using blue gradients.
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
# Define styling functions
# ========================

# Automatically determine lower_is_better and higher_is_better
lower_is_better = [
    item["name"] for item in metrics_dict.values() if item["icon"] == "(↘︎)"
]
higher_is_better = [
    item["name"] for item in metrics_dict.values() if item["icon"] == "(↗︎)"
]

# %%


def apply_styling(row):
    """Apply blue gradient and bold styling based on metric type."""
    metric = row.name
    is_lower_better = metric in lower_is_better

    # Normalize values to [0, 1] range
    if is_lower_better:
        normalized = (row.max() - row) / (row.max() - row.min())
    else:
        normalized = (row - row.min()) / (row.max() - row.min())

    # Generate styles for each cell
    styles = []
    for value, norm_value in zip(row, normalized):
        opacity = norm_value
        # Blue gradient with varying opacity
        color = f"rgba(30, 34, 170, {opacity})"  # Royal Blue

        is_best = (is_lower_better and value == row.min()) or (
            not is_lower_better and value == row.max()
        )
        font_weight = "bold" if is_best else "normal"

        style = f"background-color: {color}; font-weight: {font_weight}"
        styles.append(style)

    return styles


# %%
# Apply styling to regression metrics
# ===================================

styled_df = df.style.apply(apply_styling, axis=1).format("{:.3f}")
styled_df

# %%
# Classification task
# ===================

# %%
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

estimators = [
    make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)),
    make_pipeline(StandardScaler(), LogisticRegression()),
    RandomForestClassifier(random_state=0),
    HistGradientBoostingClassifier(random_state=0),
]

# %%
estimator_reports = [
    EstimatorReport(est, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    for est in estimators
]

# %%
comparator = ComparisonReport(reports=estimator_reports)

# %%
# Apply styling to classification metrics
# =======================================

df = comparator.metrics.report_metrics(pos_label=1, indicator_favorability=False)
styled_df = df.style.apply(apply_styling, axis=1).format("{:.3f}")
styled_df

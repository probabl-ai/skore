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
# Define styling functions
# ========================

# Automatically determine lower_is_better and higher_is_better
lower_is_better = [
    item["name"] for item in metrics_dict.values() if item["icon"] == "(↘︎)"
]
higher_is_better = [
    item["name"] for item in metrics_dict.values() if item["icon"] == "(↗︎)"
]


def apply_styling(row):
    """Apply blue gradient and bold styling based on metric type."""
    metric = row.name
    is_lower_better = metric in lower_is_better

    # Skip styling for indicator rows
    if metric.startswith("Favorability"):
        return ["" for _ in row]

    # Handle the case where we have a Favorability column
    if "Favorability" in row.index:
        # Get only numeric values, excluding the Favorability column
        numeric_row = row.drop("Favorability")
    else:
        numeric_row = row

    # Normalize values to [0, 1] range
    if is_lower_better:
        normalized = (numeric_row.max() - numeric_row) / (
            numeric_row.max() - numeric_row.min()
        )
    else:
        normalized = (numeric_row - numeric_row.min()) / (
            numeric_row.max() - numeric_row.min()
        )

    # Generate styles for each cell
    styles = []
    for value, norm_value in zip(row, normalized.reindex(row.index).fillna(0)):
        # Skip styling for non-numeric values (like indicators)
        if not isinstance(value, (int, float)):
            styles.append("")
            continue

        opacity = norm_value
        # Blue gradient with varying opacity
        color = f"rgba(30, 34, 170, {opacity})"  # Royal Blue

        is_best = (is_lower_better and value == numeric_row.min()) or (
            not is_lower_better and value == numeric_row.max()
        )
        font_weight = "bold" if is_best else "normal"

        style = f"background-color: {color}; font-weight: {font_weight}"
        styles.append(style)

    return styles


def format_values(val):
    """Format values differently based on their type."""
    if isinstance(val, (int, float)):
        return "{:.3f}".format(val)
    return str(val)


# %%
# Apply styling to regression metrics without indicators
# ======================================================

df = comparator.metrics.report_metrics(indicator_favorability=False)
print("Raw DataFrame without indicators:")
print(df)
print("\nStyled DataFrame without indicators:")
styled_df = df.style.apply(apply_styling, axis=1).format(format_values)
styled_df

# %%
# Apply styling to regression metrics with indicators
# ===================================================

df_with_indicators = comparator.metrics.report_metrics(indicator_favorability=True)
print("Raw DataFrame with indicators:")
print(df_with_indicators)
print("\nStyled DataFrame with indicators:")
styled_df_with_indicators = df_with_indicators.style.apply(
    apply_styling, axis=1
).format(format_values)
styled_df_with_indicators

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
# Apply styling to classification metrics without indicators
# ==========================================================

df = comparator.metrics.report_metrics(pos_label=1, indicator_favorability=False)
print("Raw DataFrame without indicators:")
print(df)
print("\nStyled DataFrame without indicators:")
styled_df = df.style.apply(apply_styling, axis=1).format(format_values)
styled_df

# %%
# Apply styling to classification metrics with indicators
# =======================================================

df_with_indicators = comparator.metrics.report_metrics(
    pos_label=1, indicator_favorability=True
)
print("Raw DataFrame with indicators:")
print(df_with_indicators)
print("\nStyled DataFrame with indicators:")
styled_df_with_indicators = df_with_indicators.style.apply(
    apply_styling, axis=1
).format(format_values)
styled_df_with_indicators

# %%

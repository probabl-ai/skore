"""
.. _example_feature_importance:

===============================================================
`EstimatorReport`: Get the feature importance
===============================================================

This example shows how the :class:`skore.EstimatorReport` class can be used to
quickly get insights from any scikit-learn estimator, with regards to feature importance.
"""

# %%
# Model weights
# =============

# %%
# Linear models
# ^^^^^^^^^^^^^

# %%
# Vanilla example:

# %%
from sklearn.datasets import make_regression
from skore import train_test_split, EstimatorReport
from sklearn.linear_model import LinearRegression

X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

linear_regression = LinearRegression()

linear_regression_report = EstimatorReport(
    linear_regression, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
linear_regression_report.help()

# %%
linear_regression_report.feature_importance.model_weights()

# %%
# With feature names:

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

linear_regression_report_names = EstimatorReport(
    make_pipeline(StandardScaler(), LinearRegression()),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)

linear_regression_report_names.feature_importance.model_weights()

# %%
df_model_weights = linear_regression_report_names.feature_importance.model_weights()
df_model_weights.style.bar(align="mid", color=["#d65f5f", "#5fba7d"]).format(
    precision=2
)

# %%
# .. warning::
#   a nice visualization of model weights requires scaling!

# %%
# With multi outputs:

# %%
X, y = make_regression(n_targets=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

linear_regression = LinearRegression()

linear_regression_report = EstimatorReport(
    linear_regression, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
linear_regression_report.feature_importance.model_weights()
# %%

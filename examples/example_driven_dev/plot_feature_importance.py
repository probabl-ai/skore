"""
.. _example_feature_importance:

=============================================
`EstimatorReport`: Get the feature importance
=============================================

.. warning::
    This example is meant for internal example-driven development!

This example showcases the new ``feature_importance`` accessor of the
:class:`~skore.EstimatorReport`.

Some references to have mind:

-   https://christophm.github.io/interpretable-ml-book/
-   https://scikit-learn.org/stable/auto_examples/inspection/index.html
"""

# %%
# Model weights for linear models
# ===============================

# %%
# All linear models listed in
# `scikit-learn's user guide <https://scikit-learn.org/stable/modules/linear_model.html>`_
# should work.

# %%
# Vanilla example
# ---------------

# %%
from sklearn.datasets import make_regression
from skore import train_test_split, EstimatorReport
from sklearn.linear_model import LinearRegression

X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=False)

estimator_report = EstimatorReport(
    LinearRegression(), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
estimator_report.help()

# %%
estimator_report.feature_importance.model_weights()

# %%
# .. note::
#   Does it make sense to include the ``interpret`` for model inspection?

# %%
# With feature names
# ------------------

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.head(2)

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=False)

estimator_report = EstimatorReport(
    make_pipeline(StandardScaler(), LinearRegression()),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)

estimator_report.feature_importance.model_weights()

# %%
df_model_weights = estimator_report.feature_importance.model_weights()
df_model_weights.style.bar(align="mid", color=["#d65f5f", "#5fba7d"]).format(
    precision=2
)

# %%
# .. warning::
#   When interpreting coefficients of linear models, scale matters!
#   See the following scikit-learn example:
#   `Common pitfalls in the interpretation of coefficients of linear models <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html>`_.
#   We should raise a warning for this.

# %%
# With multi outputs
# ------------------

# %%
X, y = make_regression(n_targets=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=False)

estimator_report = EstimatorReport(
    LinearRegression(), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
estimator_report.feature_importance.model_weights()

# %%
# Using another linear model
# --------------------------

# %%
from sklearn.linear_model import ElasticNet

report = EstimatorReport(
    ElasticNet(), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)
report.feature_importance.model_weights()

# %%
# .. note::
#   We might want to include unit tests for all linear models?

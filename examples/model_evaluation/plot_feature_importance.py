"""
.. _example_feature_importance:

=====================================================================
`EstimatorReport`: Inspecting your models with the feature importance
=====================================================================

As shown in :ref:`example_estimator_report`, the :class:`~skore.EstimatorReport` has
a :meth:`~skore.EstimatorReport.metrics` accessor that enables you to evaluate your
models and look at some scores that are automatically computed for you.

In practice, once you have fitted an estimator, you want to evaluate it, and
you also want to inspect it: you can do that with the estimator report's
:meth:`~skore.EstimatorReport.feature_importance` accessor that we showcase in this
example.
"""

# %%
# Linear models: inspecting the coefficients
# ==========================================

# %%
# We start by showing how skore can help you inspect the coefficients of linear models.
# We consider a linear model as defined in `scikit-learn's user guide <https://scikit-learn.org/stable/modules/linear_model.html>`_.

# %%
# Let us load some data for a regression task about predicting house prices:

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.head()

# %%
y.head()

# %%
# Now, let us apply the :class:`~skore.EstimatorReport` on a linear regression and get
# some metrics:

# %%
from skore import train_test_split, EstimatorReport
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator_report = EstimatorReport(
    LinearRegression(),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
estimator_report.metrics.report_metrics()

# %%
# Finally, we can get the coefficients of our linear model:

# %%
estimator_report.feature_importance.coefficients()

# %%
# When inspecting the coefficients of our model, we can notice that some of them
# have a very low absolute value, indicating that the corresponding features are not
# important.
# For example, that is the case of the ``HouseAge`` feature, which seems surprising.
# Indeed, we forget to scale our input data:

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

scaled_estimator_report = EstimatorReport(
    make_pipeline(StandardScaler(), LinearRegression()),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
scaled_estimator_report.feature_importance.coefficients()

# %%
# Now, the obtained coefficients can be properly interpreted.
# After scaling, it appears that the ``HouseAge`` feature is actually quite important.

# %%
# .. seealso::
#
#   For more information about the importance of scaling,
#   see scikit-learn's example on
#   `Common pitfalls in the interpretation of coefficients of linear models <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html>`_.

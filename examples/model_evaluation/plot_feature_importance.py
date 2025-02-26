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
# In short, we consider a "linear model" as a scikit-learn compatible estimator that
# holds a ``coef_`` attribute (after being fitted).

# %%
# Let us load some data for a regression task about predicting house prices:

# %%
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X.head()

# %%
y.head()

# %%
# The documentation of the California housing dataset explains that the target is
# the median house value for California districts, expressed in hundreds of thousands
# of dollars ($100,000).

# %%
# Now, let us apply the :class:`~skore.EstimatorReport` on a Ridge regression and get
# some metrics:

# %%
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator_report = EstimatorReport(
    Ridge(),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
estimator_report.metrics.report_metrics()

# %%
# From the report metrics, we have access to:
#
# - The coefficient of determination (:func:`~sklearn.metrics.r2_score`), denoted as
#   :math:`R^2`, which is a score.
#   The best possible score is :math:`1` and a constant model that always predicts the
#   average value of the target would get a score of :math:`0`.
#   Note that the score can be negative, as it could be worse than the average.
# - The root mean squared error (:func:`~sklearn.metrics.root_mean_squared_error`),
#   abbreviated as RMSE, which is an error.
#   It takes the square root of the mean squared error (MSE) so it is expressed in the
#   same units as the target variable.
#   The MSE measures the average squared difference between the predicted values and
#   the actual values.

# %%
# Finally, we can get the coefficients of our linear model:

# %%
estimator_report.feature_importance.coefficients()

# %%
# We can interpret a coefficient as follows: according to our model, on average,
# having one additional bedroom (a increase of :math:`1` of ``AveBedrms``) increases
# the house value of :math:`0.62` in $100,000, hence of $62,000.

# %%
# When further inspecting the coefficients of our model, we can notice that some of
# them have a very low absolute value, indicating that the corresponding features are
# not important.
# For example, that is the case of the ``HouseAge`` feature, which seems surprising!
# Indeed, we forget to scale our input data:

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

scaled_estimator_report = EstimatorReport(
    make_pipeline(StandardScaler(), Ridge()),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
scaled_estimator_report.feature_importance.coefficients()

# %%
# Now, after scaling, the coefficients are in the same range so they can be compared to
# one another.
# It appears that the ``HouseAge`` feature is actually quite important,
# with regards to the other coefficients.
# Hence, scaling matters for feature importance.
#
# Note that, after scaling, we can no longer interpret the coefficient values
# with regards to the original unit of the feature.

# %%
# .. seealso::
#
#   For more information about the importance of scaling,
#   see scikit-learn's example on
#   `Common pitfalls in the interpretation of coefficients of linear models <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html>`_.

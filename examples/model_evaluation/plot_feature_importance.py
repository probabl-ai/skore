"""
.. _example_feature_importance:

=====================================================================
`EstimatorReport`: Inspecting your models with the feature importance
=====================================================================

In this example, we tackle the California housing dataset where the goal is to perform
a regression task: predicting house prices based on the number of bedrooms, etc.
For that, we try out several families of models, starting with linear ones.
We evaluate these methods using skore's :class:`~skore.EstimatorReport` and its
report on metrics.

.. seealso::
    As shown in :ref:`example_estimator_report`, the :class:`~skore.EstimatorReport` has
    a :meth:`~skore.EstimatorReport.metrics` accessor that enables you to evaluate your
    models and look at some scores that are automatically computed for you.

Here, we go beyond predictive performance, and inspect these models to better interpret
their behavior, by using feature importance.
Indeed, in practice, inspection can help spot some flaws in models: it is always
recommended to look "under the hood".
For that, we use the :meth:`~skore.EstimatorReport.feature_importance` accessor of
the :class:`~skore.EstimatorReport`.
For linear models, we look at their coefficients.
"""

# %%
# Loading the dataset and performing some exploratory data analysis (EDA)
# =======================================================================

# %%
# Let us load the California housing dataset, which will enable us to perform a
# regression task about predicting house prices:

# %%
from sklearn.datasets import fetch_california_housing
import pandas as pd

X_load, y_load = fetch_california_housing(return_X_y=True, as_frame=True)
X_y = pd.concat([X_load, y_load], axis=1)
target_name = y_load.name
X_y.head(2)

# %%
# The documentation of the California housing dataset explains that the dataset
# contains aggregated data regarding each district in California in 1990 and the target
# (``MedHouseVal``) is the median house value for California districts,
# expressed in hundreds of thousands of dollars ($100,000).
# Note that there are some vacation resorts, with a large number of rooms and bedrooms.

# %%
# .. seealso::
#   For more information about the California housing dataset, refer to
#   `scikit-learn MOOC's page <https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html>`_.
#   Moreover, a more advanced modelling of this dataset is performed in
#   `this skops example <https://skops.readthedocs.io/en/stable/auto_examples/plot_california_housing.html>`_.

# %%
# Table report
# ------------

# %%
# Let us perform some quick exploration on this dataset:

# %%
from skrub import TableReport

TableReport(X_y)

# %%
# From the table report, we can draw some key observations:
#
# - Looking at the *Stats* tab, all features are numerical and there are no
#   missing values.
# - Looking at the *Distributions* tab, we can notice that some features have some
#   outliers: ``MedInc``, ``AveRooms``, ``AveBedrms``, ``Population``, and ``AveOccup``.
#   The feature with the largest number of outliers is ``AveBedrms``, probably
#   corresponding to vacation resorts.
# - Looking at the *Associations* tab, we observe that the target feature
#   ``MedHouseVal`` is mostly associated with ``MedInc``, ``Longitude``, and
#   ``Latitude``, which makes sense.

# %%
# Target feature
# --------------
#
# Moreover, the target distribution has a long tail:

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=X_y, x=target_name, bins=100)
plt.show()

# %%
# There seems to be a threshold-effect for high-valued houses: all houses with a price
# above $500,000 are given the value $500,000.

# %%

# %%
# Now, let us assess how the median income ``MedInc`` relates to the median house
# prices ``MedHouseVal``:

# %%
import plotly.express as px

X_y_plot = X_y.copy()
X_y_plot["MedInc_bins"] = pd.qcut(X_y_plot["MedInc"], q=5)
bin_order = X_y_plot["MedInc_bins"].cat.categories.sort_values()
fig = px.histogram(
    X_y_plot,
    x=target_name,
    color="MedInc_bins",
    category_orders={"MedInc_bins": bin_order},
)
fig

# %%
# As could have been expected, a high salary often comes with a more expensive house.
# We can also notice the clipping effect of house prices for very high salaries.

# %%
# Geospatial features
# -------------------

# %%
# Let us look into the coordinates of the districts in California, with regards to the
# target feature:


# %%
def plot_map(df, color_feature):
    fig = px.scatter_mapbox(
        df, lat="Latitude", lon="Longitude", color=color_feature, zoom=5, height=600
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": df["Latitude"].mean(), "lon": df["Longitude"].mean()},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    return fig


# %%
fig = plot_map(X_y, target_name)
fig

# %%
# As could be expected, the price of the houses near the sea is higher, especially
# around big cities like Los Angeles and San Jose.
# Taking into account the coordinates in our modelling will be very important.

# %%
# Splitting the data
# -----------------

# %%
# Just before diving into our first model, let us split our data into a train and a
# test split:

# %%
from sklearn.model_selection import train_test_split

X = X_y.drop(columns=[target_name])
y = X_y[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
# Linear models
# =============

# %%
# For our regression task, let us start with linear models.

# %%
# Simple model
# ------------

# %%
# Before trying any complex model, we start with a simple linear model to have a
# baseline of what a "good score" is (remember that all scores are relative).
# Here, we use a Ridge regression along with some scaling and evaluate it using
# :meth:`skore.EstimatorReport.metrics`:

# %%
from sklearn.preprocessing import StandardScaler
from skore import EstimatorReport
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ridge_report = EstimatorReport(
    make_pipeline(StandardScaler(), Ridge()),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
ridge_report.metrics.report_metrics()

# %%
# From the report metrics, let us first explain the scores we have access to:
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
#
# Here, the :math:`R^2` seems quite poor, so some further preprocessing would be needed.
# This is done further down in this example.
# For now, keep in mind that any observations drawn from inspecting the coefficients
# of this simple Ridge model are made on a model that performs quite poorly, hence
# must be treated with caution.

# %%
# To inspect our model, let us use the
# :meth:`skore.EstimatorReport.feature_importance` accessor:

# %%
ridge_report.feature_importance.coefficients()

# %%
# .. note::
#   More generally, skore can help you inspect the coefficients of all linear models.
#   We consider a linear model as defined in `scikit-learn's user guide <https://scikit-learn.org/stable/modules/linear_model.html>`_.
#   In short, we consider a "linear model" as a scikit-learn compatible estimator that
#   holds a ``coef_`` attribute (after being fitted).

# %%
# As we have performed scaling in our pipeline, the above coefficients are in the same
# range so they can be compared to one another.
# Indeed, scaling matters for feature importance.
#
# .. seealso::
#
#   For more information about the importance of scaling,
#   see scikit-learn's example on
#   `Common pitfalls in the interpretation of coefficients of linear models <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html>`_.
#
# Here, it appears that the ``MedInc``, ``Latitude``, and ``Longitude`` features are
# the most important, with regards to the absolute value of other coefficients.
# This finding is consistent with our previous observations from the *Associations*
# tab of the table report.
#
# However, due to the scaling, we can not interpret the coefficient values
# with regards to the original unit of the feature.
# Let us unscale the coefficients, without forgetting the intercerpt, so that the
# coefficients can interpreted using the original units:

# %%
import numpy as np

unscaling_weights = np.insert(ridge_report.estimator_[0].scale_, 0, 1, axis=0).reshape(
    -1, 1
)
ridge_report.feature_importance.coefficients() * unscaling_weights

# %%
# Now, we can interpret each coefficient values with regards to the original units.
# We can interpret a coefficient as follows: according to our model, on average,
# having one additional bedroom (a increase of :math:`1` of ``AveBedrms``),
# with all other features being constant,
# increases the house value of :math:`0.15` in $100,000, hence of $15,000.

# %%
# More complex model
# ------------------

# %%
# As previously mentioned, our simple Ridge model, although very easily interpretable
# with regards to the original units of the features, performs quite poorly.
# Now, we build a more complex model, with more feature engineering.
# We will see that this model will have a better score... but will be more difficult to
# interpret with regards to the original features (due to the complex feature
# engineering).

# %%
# In our previous EDA, when plotting the geospatial data with regards to the house
# prices, we noticed that it is important to take into account the latitude and
# longitude features.
# Moreover, we also observed that the median income is well associated with the
# house prices.
# Hence, we will try a feature engineering that takes into account the interactions
# of the geospatial features with features such as the income.
# The interactions are no longer simply linear as previously.

# %%
# Let us build a model with some more complex feature engineering, and still use a
# Ridge regressor (linear model) at the end.
# In particular, we perform a K-means clustering on the geospatial features:

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.compose import make_column_transformer

geo_columns = ["Latitude", "Longitude"]
spline_columns = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
]

preprocessor = make_column_transformer(
    (KMeans(n_clusters=10), geo_columns),
    remainder="passthrough",
)
engineered_ridge = make_pipeline(
    preprocessor,
    SplineTransformer(),
    PolynomialFeatures(degree=2, interaction_only=True),
    Ridge(),
)
engineered_ridge

# %%
# Now, let us compute the metrics:

# %%
engineered_ridge_report = EstimatorReport(
    engineered_ridge,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
engineered_ridge_report.metrics.report_metrics()

# %%
# We get a much better score!
# However, interpreting the features is harder: indeed, our complex feature engineering
# introduced a lot of features:

# %%
print("Initial number of features:", X_train.shape[1])

X_train_transformed = engineered_ridge_report.estimator_[:-1].transform(X_train)
print("Number of features after feature engineering:", X_train_transformed.shape[1])

# %%
# TODO
# engineered_ridge_report.feature_importance.coefficients()

# %%
# Compromising on complexity
# --------------------------

# %%
# Now, let us build a model with a more interpretable feature engineering, although
# it might perform more poorly.
# For that, after the complex feature engineering, we perform some feature selection
# using a :class:`~sklearn.feature_selection.SelectKBest`.
# To compensate the drop in score, we fine-tune some hyperparameters using
# a :class:`~sklearn.model_selection.RandomizedSearchCV`.

# %%
from scipy.stats import randint
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV

preprocessor = make_column_transformer(
    (KMeans(n_clusters=20), geo_columns),
    remainder=SplineTransformer(),
)
model = make_pipeline(
    preprocessor,
    PolynomialFeatures(degree=1, interaction_only=True),
    VarianceThreshold(),
    SelectKBest(k=50),
    Ridge(),
).set_output(transform="pandas")

parameter_grid = {
    "columntransformer__kmeans__n_clusters": randint(low=1, high=30),
    "columntransformer__remainder__degree": randint(low=1, high=4),
    "columntransformer__remainder__n_knots": randint(low=2, high=10),
    "selectkbest__k": randint(low=5, high=100),
    "ridge__alpha": np.logspace(-5, 5, num=100),
}
random_search = RandomizedSearchCV(
    model,
    param_distributions=parameter_grid,
)
random_search.fit(X_train, y_train)

compromised_model = random_search.best_estimator_
compromised_model

# %%
# Let us get the metrics for the best model of our grid search:

# %%
compromised_ridge_report = EstimatorReport(
    compromised_model,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
compromised_ridge_report.metrics.report_metrics()

# %%
# We get a good score and much less features:

# %%
print("Initial number of features:", X_train.shape[1])

X_train_transformed = compromised_ridge_report.estimator_[:-1].transform(X_train)
print("Number of features after feature engineering:", X_train_transformed.shape[1])

# %%
# Now, let us get some intuition on the results of our grid search by using a
# `plotly.express.parallel_coordinates
# <https://plotly.com/python-api-reference/generated/plotly.express.parallel_coordinates.html>`_,
# as done in
# `this scikit-learn example
# <https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html>`_:

# %%
import math


def shorten_param(param_name):
    """Remove components' prefixes in param_name."""
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


cv_results = pd.DataFrame(random_search.cv_results_)
cv_results = cv_results.rename(shorten_param, axis=1)

param_names = [shorten_param(name) for name in parameter_grid.keys()]
labels = {
    "mean_score_time": "CV Score time (s)",
    "mean_test_score": "CV score (accuracy)",
}
column_results = param_names + ["mean_test_score", "mean_score_time"]

transform_funcs = dict.fromkeys(column_results, lambda x: x)
# Using a logarithmic scale for alpha
transform_funcs["alpha"] = math.log10

fig = px.parallel_coordinates(
    cv_results[column_results].apply(transform_funcs),
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis_r,
    labels=labels,
)
fig.update_layout(
    title={
        "text": "Parallel coordinates plot of text classifier pipeline",
        "y": 0.99,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)
fig

# %%
# Using the above interactive plotly figure, one can get a better intuition of
# the impact of each hyperparameter on the CV score and the score time.

# %%
# Finally, we can visualize the results of our K-means clustering (on the training set):

# %%

# getting the cluster labels
col_transformer = compromised_model.named_steps["columntransformer"]
kmeans = col_transformer.named_transformers_["kmeans"]
clustering_labels = kmeans.labels_

# adding the cluster labels to our dataframe
X_train_plot = X_train.copy()
X_train_plot.insert(0, "clustering_labels", clustering_labels)

# plotting the map
plot_map(X_train_plot, "clustering_labels")

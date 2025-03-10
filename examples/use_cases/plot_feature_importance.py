"""
.. _example_feature_importance:

=====================================================================
`EstimatorReport`: Inspecting your models with the feature importance
=====================================================================

In this example, we tackle the California housing dataset where the goal is to perform
a regression task: predicting house prices based on features such as the number of
bedrooms, the geolocation, etc.
For that, we try out several families of models.
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
For that, we use the unified :meth:`~skore.EstimatorReport.feature_importance` accessor
of the :class:`~skore.EstimatorReport`.
For linear models, we look at their coefficients.
For tree-based models, we inspect their mean decrease in impurity (MDI).
We can also inspect the permutation feature importance, that is model-agnostic.
"""

# %%
# Loading the dataset and performing some exploratory data analysis (EDA)
# =======================================================================

# %%
# Let us load the California housing dataset, which will enable us to perform a
# regression task about predicting house prices:

# %%
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
X, y = california_housing.data, california_housing.target
california_housing.frame.head(2)

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

TableReport(california_housing.frame)

# %%
# From the table report, we can draw some key observations:
#
# - Looking at the *Stats* tab, all features are numerical and there are no
#   missing values.
# - Looking at the *Distributions* tab, we can notice that some features seem to have
#   some outliers:
#   ``MedInc``, ``AveRooms``, ``AveBedrms``, ``Population``, and ``AveOccup``.
#   The feature with the largest number of potential outliers is ``AveBedrms``, probably
#   corresponding to vacation resorts.
# - Looking at the *Associations* tab, we observe that:
#
#   -   The target feature ``MedHouseVal`` is mostly associated with ``MedInc``,
#       ``Longitude``, and ``Latitude``.
#       Indeed, intuitively, people with a large income would live in areas where the
#       house prices are high.
#       Moreover, we can expect some of these expensive areas to be close to one
#       another.
#   -   The association power between the target and these features is not super
#       high, which would indicate that each single feature can not correctly predict
#       the target.
#       Given that ``MedInc`` is associated with ``Longitude`` and also ``Latitude``,
#       it might make sense to have some interactions between these features in our
#       modelling: linear combinations might not be enough.

# %%
# Target feature
# --------------
#
# The target distribution has a long tail:

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(
    data=california_housing.frame, x=california_housing.target_names[0], bins=100
)
plt.show()

# %%
# There seems to be a threshold-effect for high-valued houses: all houses with a price
# above $500,000 are given the value $500,000.
# We keep these clipped values in our data and will inspect how our models deal with
# them.

# %%

# %%
# Now, as the median income ``MedInc`` is the feature with the highest association with
# our target, let us assess how ``MedInc`` relates to ``MedHouseVal``:

# %%
import pandas as pd
import plotly.express as px

X_y_plot = california_housing.frame.copy()
X_y_plot["MedInc_bins"] = pd.qcut(X_y_plot["MedInc"], q=5)
bin_order = X_y_plot["MedInc_bins"].cat.categories.sort_values()
fig = px.histogram(
    X_y_plot,
    x=california_housing.target_names[0],
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
# From the table report, we noticed that the geospatial features ``Latitude`` and
# ``Longitude`` were well associated with our target.
# Hence, let us look into the coordinates of the districts in California, with regards
# to the target feature, using a map:


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
fig = plot_map(california_housing.frame, california_housing.target_names[0])
fig

# %%
# As could be expected, the price of the houses near the ocean is higher, especially
# around big cities like Los Angeles, San Francisco, and San Jose.
# Taking into account the coordinates in our modelling will be very important.

# %%
# Splitting the data
# -----------------

# %%
# Just before diving into our first model, let us split our data into a train and a
# test split:

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
# Linear models: coefficients
# ===========================

# %%
# For our regression task, we first use linear models.
# For feature importance, we inspect their coefficients.

# %%
# Simple model
# ------------

# %%
# Before trying any complex feature engineering, we start with a simple pipeline to
# have a baseline of what a "good score" is (remember that all scores are relative).
# Here, we use a Ridge regression along with some scaling and evaluate it using
# :meth:`skore.EstimatorReport.metrics`:

# %%
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import EstimatorReport

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
# Let us plot the prediction error:

# %%
ridge_report.metrics.prediction_error().plot(kind="actual_vs_predicted")
plt.tight_layout()

# %%
# We can observe that the model has issues predicting large house prices, due to the
# clipping effect of the actual values.

# %%
# Now, to inspect our model, let us use the
# :meth:`skore.EstimatorReport.feature_importance` accessor:

# %%
ridge_report.feature_importance.coefficients()

# %%
# We can plot this pandas datafame:

# %%
ridge_report.feature_importance.coefficients().plot.barh(
    title="Model weights",
    xlabel="Coefficient",
    ylabel="Feature",
)
plt.tight_layout()

# %%
# .. note::
#   More generally, :meth:`skore.EstimatorReport.feature_importance` can help you
#   inspect the coefficients of all linear models.
#   We consider a linear model as defined in
#   `scikit-learn's user guide
#   <https://scikit-learn.org/stable/modules/linear_model.html>`_.
#   In short, we consider a "linear model" as a scikit-learn compatible estimator that
#   holds a ``coef_`` attribute (after being fitted).

# %%
# Since we have included scaling in the pipeline, the resulting coefficients are all on
# the same scale, making them directly comparable to each other.
# Without this scaling step, the coefficients in a linear model would be influenced by
# the original scale of the feature values, which would prevent meaningful comparisons
# between them.
#
# .. seealso::
#
#   For more information about the importance of scaling,
#   see scikit-learn's example on
#   `Common pitfalls in the interpretation of coefficients of linear models
#   <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html>`_.
#
# Here, it appears that the ``MedInc``, ``Latitude``, and ``Longitude`` features are
# the most important, with regards to the absolute value of other coefficients.
# This finding is consistent with our previous observations from the *Associations*
# tab of the table report.
#
# However, due to the scaling, we can not interpret the coefficient values
# with regards to the original unit of the feature.
# Let us unscale the coefficients, without forgetting the intercept, so that the
# coefficients can be interpreted using the original units:

# %%
import numpy as np

# retrieve the mean and standard deviation used to standardize the feature values
feature_mean = ridge_report.estimator_[0].mean_
feature_std = ridge_report.estimator_[0].scale_


def unscale_coefficients(df, feature_mean, feature_std):
    mask_intercept_column = df.index == "Intercept"
    df.loc["Intercept"] = df.loc["Intercept"] - np.sum(
        df.loc[~mask_intercept_column, "Coefficient"] * feature_mean / feature_std
    )
    df.loc[~mask_intercept_column, "Coefficient"] = (
        df.loc[~mask_intercept_column, "Coefficient"] / feature_std
    )
    return df


df_ridge_report_coef_unscaled = unscale_coefficients(
    ridge_report.feature_importance.coefficients(), feature_mean, feature_std
)
df_ridge_report_coef_unscaled

# %%
# Now, we can interpret each coefficient values with regards to the original units.
# We can interpret a coefficient as follows: according to our model, on average,
# having one additional bedroom (a increase of :math:`1` of ``AveBedrms``),
# with all other features being constant,
# increases the house value of :math:`0.62` in $100,000, hence of $62,000.
# Note that we have not dealt with any potential outlier in this iteration.

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
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

geo_columns = ["Latitude", "Longitude"]

preprocessor = make_column_transformer(
    (KMeans(n_clusters=10, random_state=0), geo_columns),
    remainder="passthrough",
)
engineered_ridge = make_pipeline(
    preprocessor,
    SplineTransformer(),
    PolynomialFeatures(degree=1, interaction_only=True, include_bias=False),
    Ridge(),
)
engineered_ridge

# %%
# Now, let us compute the metrics and compare it to our previous model using
# a :class:`skore.ComparisonReport`:

# %%
from skore import ComparisonReport

engineered_ridge_report = EstimatorReport(
    engineered_ridge,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
reports_to_compare = {
    "Vanilla Ridge": ridge_report,
    "Ridge w/ feature engineering": engineered_ridge_report,
}
comparator = ComparisonReport(reports=reports_to_compare)
comparator.metrics.report_metrics()

# %%
# We get a much better score!
# Let us plot the prediction error:

# %%
engineered_ridge_report.metrics.prediction_error().plot(kind="actual_vs_predicted")
plt.tight_layout()

# %%
# About the clipping issue, compared to the prediction error of our previous model
# (``ridge_report``), our ``engineered_ridge_report`` model seems to produce predictions
# that are not as large, so it seems that some interactions between features have
# helped alleviate the clipping issue.

# %%
# However, interpreting the features is harder: indeed, our complex feature engineering
# introduced a lot of features:

# %%
print("Initial number of features:", X_train.shape[1])

# We slice the scikit-learn pipeline to extract the predictor, using -1 to access
# the last step:
n_features_engineered = engineered_ridge_report.estimator_[-1].n_features_in_
print("Number of features after feature engineering:", n_features_engineered)

# %%
# Let us display the 15 largest absolute coefficients:

# %%
engineered_ridge_report.feature_importance.coefficients().sort_values(
    by="Coefficient", key=abs, ascending=False
).head(15).plot.barh(
    title="Model weights",
    xlabel="Coefficient",
    ylabel="Feature",
)
plt.tight_layout()

# %%
# We can observe that the most importance features are interactions between several
# features, that a simple linear model without feature engineering could not have
# captured.

# %%
# Compromising on complexity
# --------------------------

# %%
# Now, let us build a model with a more interpretable feature engineering, although
# it might not perform as well.
# For that, after the complex feature engineering, we perform some feature selection
# using a :class:`~sklearn.feature_selection.SelectKBest`.
# To compensate the drop in score, we fine-tune some hyperparameters using
# a :class:`~sklearn.model_selection.RandomizedSearchCV`.

# %%
import warnings
from scipy.stats import randint
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeCV

preprocessor = make_column_transformer(
    (KMeans(n_clusters=20, random_state=0), geo_columns),
    remainder="passthrough",
    force_int_remainder_cols=False,
)
model = make_pipeline(
    preprocessor,
    SplineTransformer(),
    PolynomialFeatures(degree=1, interaction_only=True, include_bias=False),
    VarianceThreshold(),
    SelectKBest(k=50),
    RidgeCV(np.logspace(-5, 5, num=100)),
).set_output(transform="pandas")

parameter_grid = {
    "columntransformer__kmeans__n_clusters": randint(low=10, high=30),
    "splinetransformer__degree": randint(low=1, high=4),
    "splinetransformer__n_knots": randint(low=2, high=10),
    "selectkbest__k": randint(low=5, high=100),
}

random_search = RandomizedSearchCV(
    model, param_distributions=parameter_grid, random_state=0
)

with warnings.catch_warnings():
    # Catch expected warnings to avoid cluttering this notebook,
    # especially when k > n_features
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn.feature_selection._univariate_selection",
    )
    random_search.fit(X_train, y_train)

selectkbest_ridge = random_search.best_estimator_
selectkbest_ridge

# %%
# Let us get the metrics for the best model of our grid search, and compare it with
# our previous iterations:

# %%
selectk_ridge_report = EstimatorReport(
    selectkbest_ridge,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
reports_to_compare["Ridge w/ feature selection and grid search"] = selectk_ridge_report
comparator = ComparisonReport(reports=reports_to_compare)
comparator.metrics.report_metrics()

# %%
# We get a good score and much less features:

# %%
print("Initial number of features:", X_train.shape[1])
print("Number of features after feature engineering:", n_features_engineered)

n_features_selectk = selectk_ridge_report.estimator_[-1].n_features_in_
print(
    "Number of features after feature engineering using `SelectKBest`:",
    n_features_selectk,
)

# %%
# According to the :class:`~sklearn.feature_selection.SelectKBest`, the most important
# features are the following:

# %%
selectk_features = selectk_ridge_report.estimator_[-1].feature_names_in_
print(selectk_features)

# %%
# We can see that, in the best features, according to statistical tests, there are
# geospatial features (derived from the K-means clustering) and splines on the median
# income.

# %%
# And here is the feature importance based on our model (sorted by absolute values):

# %%
selectk_ridge_report.feature_importance.coefficients().sort_values(
    by="Coefficient", key=abs, ascending=False
).head(15).plot.barh(
    title="Model weights",
    xlabel="Coefficient",
    ylabel="Feature",
)
plt.tight_layout()

# %%
# Finally, we can visualize the results of our K-means clustering (on the training set):

# %%

# getting the cluster labels
col_transformer = selectkbest_ridge.named_steps["columntransformer"]
kmeans = col_transformer.named_transformers_["kmeans"]
clustering_labels = kmeans.labels_

# adding the cluster labels to our dataframe
X_train_plot = X_train.copy()
X_train_plot.insert(0, "clustering_labels", clustering_labels)

# plotting the map
plot_map(X_train_plot, "clustering_labels")

# %%
# Tree-based models: mean decrease in impurity (MDI)
# ==================================================

# %%
# Now, let us look into tree-based models.
# For feature importance, we inspect their Mean Decrease in Impurity (MDI).
# The MDI of a feature is the normalized total reduction of the criterion (or loss)
# brought by that feature.
# The higher the MDI, the more important the feature.

# %%
# However, the MDI holds some limitations to keep in mind:
#
# - When features have large differences in cardinality, the MDI tends to favor
#   those with higher cardinality.
#   Fortunately, in this example, our numerical features share similar cardinality,
#   mitigating this concern.
# - Since MDI is typically calculated on the training set, it can reflect biases
#   from overfitting.
#   When a model overfits, the tree may partition less relevant regions of the
#   feature space, artificially inflating MDI values and distorting the perceived
#   importance of certain features.

# %%
# .. seealso::
#
#   For more information about the MDI, see scikit-learn's
#   `Permutation Importance vs Random Forest Feature Importance (MDI)
#   <https://scikit-learn.org/dev/auto_examples/inspection/plot_permutation_importance.html>`_.


# %%
# Decision trees
# --------------

# %%
# Let us start with a simple decision tree.

# %%
# .. seealso::
#   For more information about decision trees, see scikit-learn's example on
#   `Understanding the decision tree structure <https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html>`_.

# %%
from sklearn.tree import DecisionTreeRegressor

tree_report = EstimatorReport(
    DecisionTreeRegressor(random_state=0),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
reports_to_compare["Decision tree"] = tree_report

# %%
# We compare its performance with the models in our benchmark:

# %%
comparator = ComparisonReport(reports=reports_to_compare)
comparator.metrics.report_metrics()

# %%
# We note that the performance is quite poor, so the derived feature importance is to
# be dealt with caution.

# %%
# We display which accessors are available to us:

# %%
tree_report.help()

# %%
# We have a
# :meth:`~skore.EstimatorReport.feature_importance.mean_decrease_impurity`
# accessor.

# %%
# First, let us interpret our model with regards to the original features.
# For the visualization, we fix a very low ``max_depth`` so that it will be easy for
# the human eye to visualize the tree using :func:`sklearn.tree.plot_tree`:

# %%
from sklearn.tree import plot_tree

_ = plot_tree(
    tree_report.estimator_,
    feature_names=tree_report.estimator_.feature_names_in_,
    max_depth=2,
)

# %%
# This tree explains how each sample is going to be predicted by our tree.
# A decision tree provides a decision path for each sample, where the sample traverses
# the tree based on feature thresholds, and the final prediction is made at the leaf
# node (not represented above for conciseness purposes).
# At each node:
#
# - ``samples`` is the number of samples that fall into that node,
# - ``value`` is the predicted output for the samples that fall into this particular
#   node (it is the mean of the target values for the samples in that node).
#   At the root node, the value is :math:`2.074`. This means that if you were to make a
#   prediction for all :math:`15480` samples at this node (without any further splits),
#   the predicted value would be :math:`2.074`, which is the mean of the target
#   variable for those samples.
# - ``squared_error`` is the mean squared error associated with the ``value``,
#   representing the average of the squared differences between the actual target values
#   of the samples in the node and the node's predicted ``value`` (the mean),
# - the first element is how the split is defined.

# %%
# Let us explain how this works in practice.
# At each node, the tree splits the data based on a feature and a threshold.
# For the first node (the root node), ``MedInc <= 5.029`` means that, for each sample,
# our decision tree first looks at the ``MedInc`` feature (which is thus the most
# important one):
# if the ``MedInc`` value is lower than :math:`5.029` (the threshold), then the sample goes
# into the left node, otherwise it goes to the right, and so on for each node.
# As you move down the tree, the splits refine the predictions, leading to the leaf
# nodes where the final prediction for a sample is the ``value`` of the leaf it reaches.
# Note that for the second node layer, it is also the ``MedInc`` feature that is used
# for the threshold, indicating that our model heavily relies on ``MedInc``.

# %%
# .. seealso::
#   A richer display of decision trees is available in the
#   `dtreeviz <https://github.com/parrt/dtreeviz>`_ python package.
#   For example, it shows the distribution of feature values split at each node and
#   tailors the visualization to the task at hand (whether classification
#   or regression).

# %%
# Now, let us look at the feature importance based on the MDI:

# %%
tree_report.feature_importance.mean_decrease_impurity().plot.barh(
    title="Feature importance",
    xlabel="MDI",
    ylabel="Feature",
)
plt.tight_layout()

# %%
# For a decision tree, for each feature, the MDI is averaged across all splits in the
# tree. Here, the impurity is the mean squared error.
#
# As expected, ``MedInc`` is of great importance for our decision tree.
# Indeed, in the above tree visualization, ``MedInc`` is used multiple times for splits
# and contributes greatly to reducing the squared error at multiple nodes.
# At the root, it reduces the error from :math:`1.335` to :math:`0.832` and :math:`0.546`
# in the children.

# %%
# Random forest
# -------------
#
# Now, let us apply a more elaborate model: a random forest.
# A random forest is an ensemble method that builds multiple decision trees, each
# trained on a random subset of data and features.
# For regression, it averages the trees' predictions.
# This reduces overfitting and improves accuracy compared to a single decision tree.

# %%
from sklearn.ensemble import RandomForestRegressor

n_estimators = 100
rf_report = EstimatorReport(
    RandomForestRegressor(random_state=0, n_estimators=n_estimators),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
reports_to_compare["Random forest"] = rf_report

comparator = ComparisonReport(reports=reports_to_compare)
comparator.metrics.report_metrics()

# %%
# Without any feature engineering and any grid search,
# the random forest beats all linear models!

# %%
# Let us recall the number of trees in our random forest:

# %%
print(f"Number of trees in the forest: {n_estimators}")

# %%
# Given that we have many trees, it is hard to use :func:`sklearn.tree.plot_tree` as
# for the single decision tree.
# As for linear models (and the complex feature engineering), better performance often
# comes with less interpretability.

# %%
# Let us look into the MDI of our random forest:

# %%
rf_report.feature_importance.mean_decrease_impurity().plot.barh(
    title="Feature importance",
    xlabel="MDI",
    ylabel="Feature",
)
plt.tight_layout()

# %%
# In a random forest, the MDI is computed by averaging the MDI of each feature across
# all the decision trees in the forest.
#
# As for the decision tree, ``MecInc`` is the most important feature.
# Compared to linear models, the random forest also attributes a high importance to
# the ``Longitude`` and ``Latitude``, but the second most important feature is
# ``AveOccup`` that linear models did not deem very important.
# As the random forest has a great score, perhaps linear models missed something
# about ``AveOccup`` that seems to be key to predicting house prices.

# %%
# Let us visualize how ``AveOccup`` interacts with ``MedHouseVal``:

# %%
X_y_plot = california_housing.frame.copy()
X_y_plot["AveOccup"] = pd.qcut(X_y_plot["AveOccup"], q=5)
bin_order = X_y_plot["AveOccup"].cat.categories.sort_values()
fig = px.histogram(
    X_y_plot,
    x=california_housing.target_names[0],
    color="AveOccup",
    category_orders={"AveOccup": bin_order},
)
fig

# %%
# Model-agnostic: permutation feature importance
# ==============================================

# %%
# In the previous sections, we have inspected coefficients that are specific to linear
# models and the MDI that is specific to tree-based models.
# In this section, we look into the
# `permutation importance <https://scikit-learn.org/stable/modules/permutation_importance.html>`_
# which is model agnostic, meaning that it can be applied to any fitted estimator.
# In particular, it works for linear models and tree-based ones.

# %%
# Permutation feature importance measures the contribution of each feature to
# a fitted model's performance.
# It randomly shuffles the values of a single feature and observes the resulting
# degradation of the model's score.
# Permuting a predictive feature makes the performance decrease, while
# permuting a non-predictive feature does not degrade the performance much.
# By default, we compute the permutation importance on the test set.

# %%
# Now, let us look at our helper:

# %%
ridge_report.help()

# %%
# We have a :meth:`~skore.EstimatorReport.feature_importance.feature_permutation`
# accessor:

# %%
ridge_report.feature_importance.feature_permutation(random_state=0)

# %%
# The permutation importance is often calculated several times, each time
# with different permutations of the feature.
# Hence, we can have measure its variance (or standard deviation).
# Now, we plot the permutation feature importance on the train and test sets using boxplots:


# %%
def plot_permutation_train_test(est_report):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

    # Boxplot for the train set
    est_report.feature_importance.feature_permutation(
        data_source="train", random_state=0
    ).T.boxplot(ax=axes[0], vert=False)
    axes[0].set_title("Train set")
    axes[0].set_xlabel("r2")

    # Boxplot for the test set
    est_report.feature_importance.feature_permutation(
        data_source="test", random_state=0
    ).T.boxplot(ax=axes[1], vert=False)
    axes[1].set_title("Test set")
    axes[1].set_xlabel("r2")

    fig.suptitle(f"Permutation feature importance of {est_report.estimator_name_}")
    plt.tight_layout()
    plt.show()


plot_permutation_train_test(ridge_report)

# %%
# The standard deviation seems quite low.
# For both the train and test sets, the result of the inspection is the same as
# with the coefficients:
# the most important features are ``Latitude``, ``Longitude``, and ``MedInc``.

# %%
# For our decision tree, here is our permutation importance on the train and test sets:

# %%
plot_permutation_train_test(tree_report)

# %%
# The result of the inspection is the same as with the MDI:
# the most important features are ``MedInc``, ``Latitude``, ``Longitude``,
# and ``AveOccup``.

# %%
# Conclusion
# ==========
#
# In this example, we used the California housing dataset to predict house prices with
# skore's :class:`~skore.EstimatorReport`.
# By employing the :class:`~skore.EstimatorReport.feature_importance` accessor,
# we gained valuable insights into model behavior beyond mere predictive performance.
# For linear models like Ridge regression, we inspected coefficients to understand
# feature contributions, revealing the prominence of ``MedInc``, ``Latitude``,
# and ``Longitude``.
# We explained the trade-off between performance (with complex feature engineering)
# and interpretability.
# With tree-based models such as decision trees, random forests, and gradient-boosted
# trees, we utilized Mean Decrease in Impurity (MDI) to identify key features,
# notably highlighting ``AveOccup`` alongside ``MedInc``, ``Latitude``, and
# ``Longitude``.
# Compared to linear models, tree-based models rely more on ``AveOccup`` and perform
# better.
# The model-agnostic permutation feature importance further enabled us to compare
# feature significance across diverse model types.

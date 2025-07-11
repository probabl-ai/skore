"""
.. _example_sklearn_api:

=====================================================================================
Beyond traditional ML: skore's compatibility with deep learning and foundation models
=====================================================================================

This example shows how to leverage skore's capabilities with a wide variety of
scikit-learn compatible estimators: traditional machine learning frameworks as well as
deep learning frameworks and tabular foundation models.
Basically, any model that can be used with the scikit-learn API can be used with skore.
Indeed, skore's :class:`~skore.EstimatorReport` can be used to report on any estimator
that has a ``fit`` and ``predict`` method.

.. note::

  When using an :class:`~skore.EstimatorReport` to compute the ROC AUC and ROC curve
  when doing a classification task, the estimator must have a ``predict_proba`` method.

This example covers the following libraries:

- gradient boosting libraries like ``xgboost``, ``lightgbm``, and ``catboost``,
- deep learning frameworks such as ``keras`` and ``skorch`` (a wrapper for PyTorch),
- tabular foundation models such as `TabICL <https://github.com/soda-inria/tabicl>`_ and
  `TabPFN <https://github.com/PriorLabs/TabPFN>`_,
- time series classification with
  `tslearn <https://tslearn.readthedocs.io/en/stable/>`_
  and `aeon <https://github.com/aeon-toolkit/aeon>`_,
- time series forecasting with a cross-validation using a time series splitter.

.. note::

  This example is not exhaustive and many more libraries are supported!
"""

# %%
# Binary classification on tabular data
# =====================================
#
# Loading a binary classification dataset
# ---------------------------------------
#
# We generate a synthetic binary classification dataset with only 1,000 samples to keep
# the computation time reasonable, especially for the tabular foundation models:

# %%
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1_000,
    n_features=5,
    n_classes=2,
    class_sep=0.3,
    n_clusters_per_class=1,
    random_state=42,
)
print(f"{X.shape = }")

# %%
# We split our data:

# %%
from skore import train_test_split

split_data = train_test_split(X, y, random_state=42, as_dict=True)

# %%
# Tree-based models
# -----------------
#
# For this binary classification task, the first family of models we shall consider
# are tree-based models from libraries external to scikit-learn.

# %%
# XGBoost
# ^^^^^^^

# %%
from skore import EstimatorReport
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)

xgb_report = EstimatorReport(xgb, pos_label=1, **split_data)
xgb_report.metrics.summarize().frame()

# %%
# We can easily get the summary of metrics, and also a ROC curve plot for example:

# %%
xgb_report.metrics.roc().plot()

# %%
# LightGBM
# ^^^^^^^^

# %%
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=0
)

lgbm_report = EstimatorReport(lgbm, pos_label=1, **split_data)
lgbm_report.metrics.summarize().frame()

# %%
# CatBoost
# ^^^^^^^^

# %%
from catboost import CatBoostClassifier

catboost = CatBoostClassifier(
    iterations=50,
    depth=3,
    learning_rate=0.1,
    random_state=42,
    verbose=False,
    allow_writing_files=False,
)

catboost_report = EstimatorReport(catboost, pos_label=1, **split_data)
catboost_report.metrics.summarize().frame()

# %%
# Deep learning with neural networks
# ----------------------------------
#
# It is often said that scikit-learn is mainly for traditional machine learning.
# However, if the neural network is an estimator compatible with the scikit-learn API,
# such as Keras or PyTorch (through skorch), they are compatible with a scikit-learn
# grid search for example.
# Moreover, it means that we can use skore to report on it.
# Hence, skore is not limited to traditional machine learning models.

# %%
# Keras
# ^^^^^
#
# `Keras <https://github.com/keras-team/keras>`_ is a multi-backend deep learning
# framework that enables us to work with JAX, TensorFlow, and PyTorch.
# Since the following pull request https://github.com/keras-team/keras/pull/20599
# merged in December 2024, scikit-learn wrappers are back in Keras, meaning that we can
# use Keras models directly with skore.

# %%
from keras.src.layers import Dense, Input
from keras.src.models import Model
from keras.src.wrappers import SKLearnClassifier


def create_mlp_classifier(X, y, loss, layers=[10]):
    """Creates a basic MLP classifier for binary classification."""
    n_features_in = X.shape[1]
    inp = Input(shape=(n_features_in,))

    hidden = inp
    for layer_size in layers:
        hidden = Dense(layer_size, activation="relu")(hidden)

    # Binary classification with one-hot encoding: 2 outputs with softmax activation
    out = Dense(2, activation="softmax")(hidden)
    model = Model(inp, out)
    model.compile(loss=loss, optimizer="rmsprop")

    return model


keras_model = SKLearnClassifier(
    model=create_mlp_classifier,
    model_kwargs={
        "loss": "categorical_crossentropy",
        "layers": [32],
    },
    fit_kwargs={"epochs": 5, "verbose": 0},
)

# %%
keras_report = EstimatorReport(keras_model, pos_label=1, **split_data)
print(keras_report.metrics.precision())

# %%
# .. note::
#
#   The estimator above does not have a `predict_proba` method, so we can not compute
#   the ROC AUC nor the ROC curve; and thus not the summary of metrics.
#   Also note that users could implement a `predict_proba` method for this model by
#   wrapping it.

# %%
keras_report.metrics.accuracy()

# %%
# PyTorch using skorch
# ^^^^^^^^^^^^^^^^^^^^
#
# We shall create our neural network model using PyTorch.
# We consider a neural network with 2 hidden layers and 1 output layer with ReLU
# activations.

# %%
from torch import nn


class MyNeuralNet(nn.Module):
    def __init__(self, input_dim=5, h1=64, h2=32, output_dim=2):
        super(MyNeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, X):
        return self.layers(X)


# %%
# Since we want to use this with `skorch` that provides a sklearn like API interface
# that `skore` can utilize, we shall wrap this in `skorch`'s ``NeuralNetClassifier``.

# %%
import torch
from skorch import NeuralNetClassifier

torch.manual_seed(42)

skorch_model = NeuralNetClassifier(
    MyNeuralNet,
    max_epochs=10,
    lr=1e-2,
    criterion=torch.nn.CrossEntropyLoss,
    classes=list(range(2)),
)

# %%
# Next, we create our final model by wrapping this into a scikit-learn pipeline that
# adds a standard scaler.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# Create a transformer to convert data to float32 for PyTorch compatibility
# This is necessary because scikit-learn data is typically float64, but PyTorch defaults to float32
class Float32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(np.float32)


skorch_pipeline = make_pipeline(StandardScaler(), Float32Transformer(), skorch_model)
skorch_pipeline.fit(split_data["X_train"], split_data["y_train"])

# %%
from skore import EstimatorReport

skorch_report = EstimatorReport(
    skorch_pipeline,
    fit=True,
    pos_label=1,
    **split_data,
)

# %%
# Similar to the above, we can observe the report and the ROC curves of our final model.

# %%
skorch_report.metrics.roc().plot()
skorch_report.metrics.summarize(indicator_favorability=True).frame()


# %%
# Tabular foundation models
# -------------------------

# %%
# TabICL
# ^^^^^^

# %%
from tabicl import TabICLClassifier

tabicl = TabICLClassifier()
tabicl_report = EstimatorReport(tabicl, pos_label=1, **split_data)
tabicl_report.metrics.summarize().frame()

# %%
# TabPFN
# ^^^^^^

# %%
from tabpfn import TabPFNClassifier

tabpfn = TabPFNClassifier()
tabpfn_report = EstimatorReport(tabpfn, pos_label=1, **split_data)
tabpfn_report.metrics.summarize().frame()

# %%
# Custom model
# ------------

# %%
# Let us use a custom estimator inspired from the
# `scikit-learn documentation <https://scikit-learn.org/dev/developers/develop.html#rolling-your-own-estimator>`_,
# a nearest neighbor classifier:

# %%
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class CustomClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


# %%
# .. note::
#
#   The estimator above does not have a `predict_proba` method.

# %%
# We can now use this model with skore:

# %%
template_report = EstimatorReport(CustomClassifier(), pos_label=1, **split_data)
template_report.metrics.precision()

# %%
# Benchmark of all of the above models
# ------------------------------------

# %%
# .. note::
#
#   `keras_report` and `template_report` do not have a `predict_proba` method, so we
#   do not include them in the comparison report.

# %%
from skore import ComparisonReport

estimators = [
    xgb_report,
    lgbm_report,
    catboost_report,
    skorch_report,
    tabicl_report,
    tabpfn_report,
]

comparator = ComparisonReport(
    estimators,
)
comparator.metrics.summarize().frame()

# %%
comparator.metrics.roc().plot()

# %%
# Time series classification
# ==========================

# %%
# Load a time series dataset
# --------------------------

# %%
from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax

np.random.seed(0)
n_ts_per_blob, sz, d, n_blobs = 20, 100, 1, 2

# Prepare data
X, y = random_walk_blobs(n_ts_per_blob=n_ts_per_blob, sz=sz, d=d, n_blobs=n_blobs)
scaler = TimeSeriesScalerMinMax(value_range=(0.0, 1.0))  # Rescale time series
X_scaled = scaler.fit_transform(X)

# %%
print(f"X_scaled.shape: {X_scaled.shape}")
print(f"y.shape: {y.shape}")

# %%
split_data = train_test_split(X_scaled, y, random_state=42, as_dict=True)

# %%
# .. note::
#
#   We have a dataset of several time series, and we want to classify them.

# %%
# tslearn
# -------

# %%
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

ts_model = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
ts_model_report = EstimatorReport(ts_model, pos_label=1, **split_data)
ts_model_report.metrics.summarize().frame()

# %%
# aeon
# ----

# %%
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

aeon_model = KNeighborsTimeSeriesClassifier(n_neighbors=1)
aeon_model_report = EstimatorReport(aeon_model, pos_label=1, **split_data)
aeon_model_report.metrics.summarize().frame()

# %%
# Time series forecasting with cross-validation
# =============================================
#
# The goal of this section is to show that a :class:`skore.CrossValidationReport`
# can be used with a :class:`sklearn.model_selection.TimeSeriesSplit` when performing
# time series forecasting.

# %%
# Let us take inspiration from the `Time-related feature engineering <https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html>`_ example of the scikit-learn documentation.
#
# We shall use the bike sharing demand dataset:

# %%
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
df = bike_sharing.frame

# %%
# As done in the scikit-learn example, we shall preprocess the data:

# %%
y = df["count"] / df["count"].max()
X = df.drop("count", axis="columns")

X["weather"] = (
    X["weather"]
    .astype(object)
    .replace(to_replace="heavy_rain", value="rain")
    .astype("category")
)

# %%
# We will use time-based cross-validation strategy to evaluate our model:

# %%
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10_000,
    test_size=1_000,
)

# %%
# We define our model:

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

gbrt = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)

# %%
# We create our skore cross-validation report:

# %%
from skore import CrossValidationReport

cv_report = CrossValidationReport(gbrt, X, y, cv_splitter=ts_cv)
cv_report.metrics.summarize().frame()

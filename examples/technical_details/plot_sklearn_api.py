"""
.. _example_sklearn_api:

===========================================================
Using skore for models compatible with the scikit-learn API
===========================================================

This example shows how to leverage skore's capabilities with scikit-learn compatible
estimators, including:

- libraries like ``xgboost``, ``catboost``, and ``lightgbm``,
- deep learning frameworks such as ``skorch`` (a wrapper for PyTorch) and ``keras``,
- tabular foundation models such as `TabICL <https://github.com/soda-inria/tabicl>`_ and
  `TabPFN <https://github.com/PriorLabs/TabPFN>`_,
- models specific to time series data such as
  `tslearn <https://tslearn.readthedocs.io/en/stable/>`_,
  `mlforecast <https://github.com/Nixtla/mlforecast>`_,
  and `aeon <https://github.com/aeon-toolkit/aeon>`_.
"""

# %%
# Binary classification on tabular data
# =====================================
#
# Loading a binary classification dataset
# ---------------------------------------

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
from skore import train_test_split

split_data = train_test_split(X, y, random_state=42, as_dict=True)

# %%
# Tree-based models
# -----------------

# %%
# XGBoost
# ^^^^^^^

# %%
from skore import EstimatorReport
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, random_state=42)

xgb_report = EstimatorReport(xgb, pos_label=1, **split_data)
xgb_report.metrics.summarize().frame()

# %%
# Deep learning with neural networks
# ----------------------------------
#
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
# Let us use the `TemplateClassifier` from the
# `scikit-learn documentation <https://scikit-learn.org/dev/developers/develop.html>`_ which is a nearest neighbor classifier, by adding the `predict_proba` method:

# %%
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class TemplateClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

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

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        predictions = self.y_[closest]

        # Create probability matrix
        proba = np.zeros((len(X), len(self.classes_)))
        for i, pred in enumerate(predictions):
            class_idx = np.where(self.classes_ == pred)[0][0]
            proba[i, class_idx] = 1.0

        return proba


# %%
# We can now use this model with skore:

# %%
template_report = EstimatorReport(TemplateClassifier(), pos_label=1, **split_data)
template_report.metrics.summarize().frame()

# %%
# Benchmark of all of the above models
# ------------------------------------

# %%
from skore import ComparisonReport

estimators = [xgb_report, skorch_report, tabicl_report, tabpfn_report, template_report]

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

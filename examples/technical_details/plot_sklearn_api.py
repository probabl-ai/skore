"""
.. _example_sklearn_api:

===========================================================
Using skore for models compatible with the scikit-learn API
===========================================================
"""

# %%
# Loading the data
# ================

# %%
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
X.head()

# %%
from skore import train_test_split

split_data = train_test_split(X, y, random_state=42, as_dict=True)

# %%
# Tree-based models
# =================

# %%
# XGBoost
# -------

# %%
from skore import EstimatorReport
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, random_state=42)

xgb_report = EstimatorReport(xgb, pos_label=1, **split_data)
xgb_report.metrics.summarize().frame()

# %%
# Neural networks
# ===============

# %%
# Pytorch using skorch
# --------------------

# %%
from torch import nn


class MyNeuralNet(nn.Module):
    def __init__(self, input_dim=30, h1=64, h2=32, output_dim=2):
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
import torch
from skorch import NeuralNetClassifier

torch.manual_seed(42)

nnet = NeuralNetClassifier(
    MyNeuralNet,
    max_epochs=10,
    lr=1e-2,
    criterion=torch.nn.CrossEntropyLoss,
    classes=list(range(2)),
)

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

skorch_nn = make_pipeline(StandardScaler(), nnet)
skorch_nn

# %%
if False:
    skorch_nn_report = EstimatorReport(skorch_nn, pos_label=1, **split_data)
    skorch_nn_report.metrics.summarize().frame()

# %%
# Tabular foundation models
# =========================

# %%
# TabICL
# ------

# %%
from tabicl import TabICLClassifier

tabicl = TabICLClassifier()
tabicl_report = EstimatorReport(tabicl, pos_label=1, **split_data)
tabicl_report.metrics.summarize().frame()

# %%
# TabPFN
# ------

# %%
from tabpfn import TabPFNClassifier

tabpfn = TabPFNClassifier()
tabpfn_report = EstimatorReport(tabpfn, pos_label=1, **split_data)
tabpfn_report.metrics.summarize().frame()

# %%
# Time series models
# ==================

# %%
# Load a time series dataset
# --------------------------

# %%
from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax
import numpy as np

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

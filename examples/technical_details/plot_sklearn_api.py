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
# Loading a dataset, performing some analysis, and some preprocessing
# ==================================================================
#
# Loading a binary classification dataset
# ---------------------------------------
#
# In this example, we tackle the Breast Cancer Winconsin dataset which is a
# binary classification task, i.e. predicting whether a tumor is malignant or benign.
# We choose this dataset to keep this example simple.

# %%
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print(f"{X.shape = }")
X.head(2)

# %%
# The `documentation <https://sklearn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>`_
# shows that there are 2 classes `malignant` (M) and `benign` (B) with a slight
# imbalance 212 `M` samples and 357 `B` samples.
# Before we approach the balancing problem, let us explore the dataset.
#
# Exploratory data analysis
# -------------------------
#
# We shall explore the dataset using `skrub` library's :class:`~skrub.TableReport`:

# %%
from skrub import TableReport

TableReport(X)

# %%
import pandas as pd

pd.DataFrame(y).value_counts(normalize=True).round(2)

# %%
# Preprocessing
# -------------

# %%
# From the table report, we can make a few inferences:
#
# - The *Stats* tab shows we only have numerical values and that there are no null
#   values.
# - The *Distribution* tab shows us there is moderate level of imbalance: 63% benign
#   from the mean value and 37% malignant values. While we can balance this or add
#   class weights in our neural network, it is important to note that we're modeling
#   the real world and not to achieve an artificial balance!
# - The *Distribution* tab shows a few features that have some outliers, namely:
#   ``radius error``, ``texture error``, ``perimeter error``, ``area error``,
#   ``smoothness error``, ``compactness error``, ``concavity error``,
#   ``concave points error``, ``symmetry error``, ``fractal dimension error``,
#   ``worst area``, ``worst symmetry``, ``worst fractal dimensions``.
#   However, it is important to note that the outliers range from 1-8 in all the
#   different columns which is not huge to cause problems in our modeling.
# - The *Association* tab shows that a table with correlation analysis between the different features. We can infer a few things from this as per below:
#
#   - We can select features that show a strong association with our target.
#   - Since we're using a deep learning example, multicollinearity is less of a concern compared to linear models as they can handle correlated features through their hidden layers and non-linear transformations. However, we can remove a few redundant features which can improve efficiency, faster convergence and interpretability.
#   - Some examples of reasoning can be seen below.
#   - mean radius and mean perimeter is one example. They convey the same information and can be removed.
#   - Similarly, we could remove mathematically related features, and keep the one which is most correlated. For e.g., among mean radius, mean perimeter, and mean area, we could pick one instead of keeping all three.
#   - We can see that there are no direct correlations between error measurements and the target and hence we skip them.

# %%
import numpy as np

cols_to_keep = [
    "worst perimeter",
    "mean radius",
    "worst concave points",
    "mean concave points",
    "worst concavity",
    "mean concavity",
]

X = X[cols_to_keep]
X_numpy = X.values.astype(np.float32)

# %%
# Splitting the data
# ------------------
#
# We shall split the data using `skore`'s `train_test_split`.

# %%
from skore import train_test_split

split_data = train_test_split(X_numpy, y, random_state=42, as_dict=True)

# %%
# We can see how `skore` gives us 2 warnings on split to help us with our modeling approach.
#
# - Adhering to the `HighClassImbalanceTooFewExamplesWarning`, we shall employ cross validation in our modeling approach.
# - The `ShuffleTrueWarning` can be ignored as there are no temporal dependencies in our medical dataset. Each example is IID.

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
# Deep learning with neural networks
# ==================================
#
# PyTorch using skorch
# --------------------
#
# We shall create our neural network model using PyTorch.
# We consider a neural network with 2 hidden layers and 1 output layer with ReLU
# activations.

# %%
from torch import nn


class MyNeuralNet(nn.Module):
    def __init__(self, input_dim=6, h1=64, h2=32, output_dim=2):
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

skorch_pipeline = make_pipeline(StandardScaler(), skorch_model)
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
# Benchmark
# =========

# %%
from skore import ComparisonReport

estimators = [xgb_report, skorch_report, tabicl_report, tabpfn_report]

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
    max_train_size=10000,
    test_size=1000,
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

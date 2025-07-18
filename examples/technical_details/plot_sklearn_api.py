"""
.. _example_sklearn_api:

=====================================================================================
Beyond traditional ML: skore's compatibility with deep learning and foundation models
=====================================================================================

This example shows how to leverage skore's capabilities with a wide variety of
scikit-learn compatible estimators: from traditional machine learning frameworks
such as gradient boosting to cutting-edge deep learning frameworks and tabular
foundation models.

For all these kinds of models, skore provides a unified interface for consistent reporting, visualization, and comparison.
Basically, any model that can be used with the scikit-learn API can be used with skore.
Indeed, skore's :class:`~skore.EstimatorReport` can be used to report on any estimator
that has a ``fit`` and ``predict`` method.

.. note::

  When using an :class:`~skore.EstimatorReport` to compute the ROC AUC and ROC curve
  when doing a classification task, the estimator must have a ``predict_proba`` method.

This example covers the following libraries:

- gradient boosting libraries like `XGBoost <https://github.com/dmlc/xgboost>`_,
  `LightGBM <https://github.com/microsoft/LightGBM>`_, and
  `CatBoost <https://github.com/catboost/catboost>`_.
- deep learning frameworks such as `Keras <https://github.com/keras-team/keras>`_ and
  `skorch <https://github.com/skorch-dev/skorch>`_
  (a wrapper for `PyTorch <https://github.com/pytorch/pytorch>`_),
- `pytabkit <https://github.com/dholzmueller/pytabkit>`_ which provides
  scikit-learn interfaces for modern tabular classification and regression methods,
- tabular foundation models such as `TabICL <https://github.com/soda-inria/tabicl>`_ and
  `TabPFN <https://github.com/PriorLabs/TabPFN>`_,
- time series classification with
  `tslearn <https://github.com/tslearn-team/tslearn>`_
  and `aeon <https://github.com/aeon-toolkit/aeon>`_,
- time series forecasting with a skore cross-validation
  (:class:`~skore.CrossValidationReport`) using a time series splitter
  (:class:`~sklearn.model_selection.TimeSeriesSplit`).

.. note::

  This example is not exhaustive and many more libraries and frameworks are supported!
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
# Gradient-boosted decision trees models
# --------------------------------------
#
# For this binary classification task, the first family of models we shall consider
# are gradient-boosted decision trees models from libraries external to scikit-learn.
# The most popular are `XGBoost <https://github.com/dmlc/xgboost>`_,
# `LightGBM <https://github.com/microsoft/LightGBM>`_, and
# `CatBoost <https://github.com/catboost/catboost>`_.

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
# We filter out the warning about the feature names, which is not relevant for this example:

# %%
from lightgbm import LGBMClassifier
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        action="ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        category=UserWarning,
        module="sklearn",
    )

    lgbm = LGBMClassifier(
        n_estimators=20,
        max_depth=2,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )

    lgbm_report = EstimatorReport(lgbm, pos_label=1, **split_data)
    df_lgbm_metrics = lgbm_report.metrics.summarize().frame()

df_lgbm_metrics

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
# Since the https://github.com/keras-team/keras/pull/20599 pull request merged in
# December 2024, scikit-learn wrappers are back in Keras, meaning that we can use Keras
# models directly with skore.

# %%
from keras.src.layers import Dense, Input
from keras.src.models import Model
from keras.src.wrappers import SKLearnClassifier


def create_mlp_classifier(X, y, loss, layers=[10]):
    n_features_in = X.shape[1]
    inp = Input(shape=(n_features_in,))

    hidden = inp
    for layer_size in layers:
        hidden = Dense(layer_size, activation="relu")(hidden)

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
# We can now use skore's :class:`~skore.EstimatorReport`:

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
# `skorch <https://github.com/skorch-dev/skorch>`_ is a library that wraps PyTorch
# models to make them compatible with the scikit-learn API.
# For that, we use a `skorch`'s ``NeuralNetClassifier`` wrapper.
# We also add a standard scaler and a transformer to convert the data to float32.

# %%
from torch import nn
from torch import manual_seed
from skorch import NeuralNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

manual_seed(42)


class SimpleNeuralNet(nn.Module):
    """A simple PyTorch neural network for binary classification."""

    def __init__(self, input_dim=5, h1=64, h2=32, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, X):
        return self.layers(X)


class Float32Transformer(BaseEstimator, TransformerMixin):
    """As scikit-learn data is typically float64, but PyTorch defaults to float32,
    we need to convert the data to float32.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(np.float32)


skorch_model = make_pipeline(
    StandardScaler(),
    Float32Transformer(),
    NeuralNetClassifier(
        SimpleNeuralNet,
        max_epochs=10,
        lr=1e-2,
        criterion=nn.CrossEntropyLoss,
        classes=list(range(2)),
        verbose=0,
    ),
)
skorch_model.fit(split_data["X_train"], split_data["y_train"])

# %%
# We can now use skore to report on the model:

# %%
skorch_report = EstimatorReport(
    skorch_model,
    fit=True,
    pos_label=1,
    **split_data,
)
skorch_report.metrics.summarize().frame()

# %%
# RealMLP
# ^^^^^^^
#
# Next, we use the `pytabkit <https://github.com/dholzmueller/pytabkit>`_ library that
# provides scikit-learn interfaces for modern tabular classification and regression
# methods that are benchmarked.
# First, we start with the RealMLP model which is novel neural net model with tuned
# defaults (TD) introduced in the
# `Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data <https://arxiv.org/pdf/2407.04491>`_ paper.
# RealMLP is an improved tuned multi-layer perceptron (MLP) that is optimized for
# tabular data.

# %%
from pytabkit import RealMLP_TD_Classifier

realmlp = RealMLP_TD_Classifier(random_state=42)
realmlp_report = EstimatorReport(realmlp, pos_label=1, **split_data)
realmlp_report.metrics.summarize().frame()

# %%
# TabM
# ^^^^
# Next, we use the TabM model from
# `TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling <https://arxiv.org/abs/2410.24210>`_.

# %%
from pytabkit import TabM_D_Classifier

tabm = TabM_D_Classifier(random_state=42)
tabm_report = EstimatorReport(tabm, pos_label=1, **split_data)
tabm_report.metrics.summarize().frame()

# %%
# Tabular foundation models
# -------------------------

# %%
# TabPFN
# ^^^^^^
#
# TabPFN is a foundation model for tabular data that is based on the
# `Accurate predictions on small data with a tabular foundation model <https://www.nature.com/articles/s41586-024-08328-6>`_ paper.
# It is said to outperform gradient-boosted decision trees on datasets with up to
# 10k samples by a wide margin.
# It works for classification and regression.

# %%
from tabpfn import TabPFNClassifier

tabpfn = TabPFNClassifier(random_state=42)
tabpfn_report = EstimatorReport(tabpfn, pos_label=1, **split_data)
tabpfn_report.metrics.summarize().frame()

# %%
# TabICL
# ^^^^^^
#
# TabICL is a tabular foundation model that is based on the
# `TabICL: A Tabular Foundation Model for In-Context Learning on Large Data <https://arxiv.org/pdf/2502.05564>`_ paper.
# It is made for classification, pre-trained on synthetic datasets with up to 60k
# samples and capable of handling 500k samples on affordable resources.
# It is said to be faster than TabPFNv2.

# %%
from tabicl import TabICLClassifier

tabicl = TabICLClassifier(random_state=42)
tabicl_report = EstimatorReport(tabicl, pos_label=1, **split_data)
tabicl_report.metrics.summarize().frame()

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
custom_report = EstimatorReport(CustomClassifier(), pos_label=1, **split_data)
custom_report.metrics.precision()

# %%
# Benchmark of all of the above models
# ------------------------------------

# %%
# .. note::
#
#   `keras_report` and `custom_report` do not have a `predict_proba` method, so we
#   do not include them in the following comparison report.

# %%
from skore import ComparisonReport

estimators = [
    xgb_report,
    lgbm_report,
    catboost_report,
    skorch_report,
    realmlp_report,
    tabm_report,
    tabpfn_report,
    tabicl_report,
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
# Now, we no longer consider the binary classification task with the dataset generated
# at the beginning of this example, but a time series classification task.

# %%
# tslearn
# -------
#
# `tslearn <https://github.com/tslearn-team/tslearn>`_ is a Python package that
# provides machine learning tools for the analysis of time series.
# This package builds on scikit-learn, numpy and scipy libraries.
# Note that it does not implement any forecasting methods.

# %%
# Let us generate a synthetic time series dataset for classification:

# %%
from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax

np.random.seed(0)
n_ts_per_blob, sz, d, n_blobs = 20, 100, 1, 2

X, y = random_walk_blobs(n_ts_per_blob=n_ts_per_blob, sz=sz, d=d, n_blobs=n_blobs)
scaler = TimeSeriesScalerMinMax(value_range=(0.0, 1.0))
X_scaled = scaler.fit_transform(X)

# %%
print(f"X_scaled.shape: {X_scaled.shape}")
print(f"y.shape: {y.shape}")

# %%
# We have 40 univariate time series (of dimension 1) each of length 100.
# More information about the tslearn time series format can be found in the
# `tslearn documentation <https://tslearn.readthedocs.io/en/stable/gettingstarted.html#time-series-format>`_.

# %%
# We split our data:

# %%
split_data = train_test_split(X_scaled, y, random_state=42, as_dict=True)

# %%
print(f"{split_data['X_train'].shape = }")
print(f"{split_data['X_test'].shape = }")

# %%
# .. note::
#
#   We have a dataset of several time series, and we want to classify them.
#   In particular, we do not need a :class:`~sklearn.model_selection.TimeSeriesSplit`.

# %%
# We use a nearest neighbor classifier from tslearn using the
# `Dynamic Time Warping (DTW) <https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html>`_
# distance:

# %%
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

ts_model = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
ts_model_report = EstimatorReport(ts_model, pos_label=1, **split_data)
ts_model_report.metrics.summarize().frame()

# %%
# .. note::
#
#   When computing the distance matrix in the nearest neighbor algorithm, the Euclidean
#   distance is often used by default.
#   However, the DTW distance is often better suited for time series data as it is more
#   robust to time shifts.
#   The DTW distance is specific to time series data and is not implemented in
#   scikit-learn but in tslearn.

# %%
# aeon
# ----
#
# `aeon <https://github.com/aeon-toolkit/aeon>`_ is a toolkit for machine learning
# with time series data which is compatible with scikit-learn.
# It implements state-of-the-art methods in a wide variety of time series tasks
# including anomaly detection, classification, and forecasting.

# %%
# Let us load a real-world time series classification dataset from aeon, which comes
# with a native train-test split:

# %%
from aeon.datasets import load_arrow_head

arrow, arrow_labels = load_arrow_head(split="train")
print(f"ArrowHead series of type {type(arrow)} and shape {arrow.shape}")

# %%
# .. note::
#
#   aeon has its own specific time series format, which is not compatible with tslearn.
#   Its `documentation <https://www.aeon-toolkit.org/en/stable/examples/classification/classification.html#Data-Storage-and-Problem-Types>`_
#   states that their format is ``(n_cases, n_channels, n_timepoints)``,
#   where ``n_cases`` is the number of time series, ``n_channels`` is the dimension,
#   and ``n_timepoints`` is the number of time points.

# %%
# We load the test set:

# %%
arrow_test, arrow_test_labels = load_arrow_head(split="test", return_type="numpy2d")

# %%
# We use a :class:`~aeon.classification.convolution_based.RocketClassifier` from aeon,
# which is a `convolution based classifier <https://www.aeon-toolkit.org/en/stable/examples/classification/convolution_based.html>`_:

# %%
from aeon.classification.convolution_based import RocketClassifier

rocket = RocketClassifier(n_kernels=2_000)
arrow2d = arrow.squeeze()
rocket_report = EstimatorReport(
    rocket,
    X_train=arrow2d,
    y_train=arrow_labels,
    X_test=arrow_test,
    y_test=arrow_test_labels,
)
rocket_report.metrics.summarize().frame()

# %%
# Time series forecasting with cross-validation
# =============================================
#
# Now, we no longer use the time series classification dataset: we will focus on a
# time series forecasting task.
#
# The goal of this section is to show that a :class:`skore.CrossValidationReport`
# can be used with a :class:`sklearn.model_selection.TimeSeriesSplit` when performing
# time series forecasting.

# %%
# Let us take inspiration from the `Time-related feature engineering <https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html>`_ example of the scikit-learn documentation.
#
# We shall use the bike sharing demand dataset whose goal is to predict the hourly
# demand:

# %%
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
df = bike_sharing.frame

# %%
# As explained in the scikit-learn example, we shall preprocess the data by rescaling
# the target variable and replace the weather category ``heavy_rain`` by ``rain`` as
# there are too few instances for ``heavy_rain``:

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
# We will use time-based cross-validation strategy to evaluate our model with a gap of
# 48 hours between train and test and 1,000 test datapoints which should be enough:

# %%
from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=48,
    max_train_size=10_000,
    test_size=1_000,
)

# %%
# We define our model which is a modern scikit-learn
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
# with native support for categorical features:

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

gbrt = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)

# %%
# We evaluate our model thanks to a :class:`~skore.CrossValidationReport`:

# %%
from skore import CrossValidationReport

cv_report = CrossValidationReport(gbrt, X, y, cv_splitter=ts_cv)
cv_report.metrics.summarize().frame()

# %%
# We can also investigate the metrics for each fold:

# %%
cv_report.metrics.summarize(aggregate=None).frame()

# %%
# Conclusion
# ==========
#
# This example demonstrates skore's remarkable flexibility and compatibility with a wide
# range of machine learning frameworks and models.
# The key takeaway is that any estimator that follows the scikit-learn API can be
# seamlessly integrated with skore.
#
# We have shown that skore works with:
#
# - Traditional ML: Gradient boosting libraries (XGBoost, LightGBM, CatBoost)
# - Deep learning: Neural networks via Keras and PyTorch (through skorch)
# - Modern tabular models: RealMLP, TabM, and foundation models like TabPFN and TabICL
# - Time series: Specialized libraries like tslearn and aeon
# - Custom models: Any estimator implementing the scikit-learn interface
#
# This universality makes skore a powerful tool for model evaluation and comparison
# across different ML paradigms, enabling practitioners to use consistent reporting
# and visualization tools.
#
# .. seealso::
#
#   For a practical example of using language models within scikit-learn pipelines,
#   see :ref:`example_use_case_employee_salaries` which demonstrates how to use
#   skrub's :class:`~skrub.TextEncoder` (a language model-based encoder) in a
#   scikit-learn pipeline for feature engineering.
#
# .. seealso::
#
#   For an example of wrapping Large Language Models (LLMs) to be compatible with
#   scikit-learn APIs, see the tutorial on `Quantifying LLMs Uncertainty with Conformal
#   Predictions <https://medium.com/capgemini-invent-lab/quantifying-llms-uncertainty-with-conformal-predictions-567870e63e00>`_.
#   The article demonstrates how to wrap models like Mistral-7B-Instruct in a
#   scikit-learn-compatible interface.

"""
.. _example_sklearn_api:

===========================================================
Using skore for models compatible with the scikit-learn API
===========================================================

This example shows how to leverage skore's capabilities with scikit-learn compatible
estimators, including:

- libraries like ``xgboost``, ``catboost``, and ``lightgbm``,
- deep learning frameworks such as ``skorch`` (a wrapper for PyTorch) and ``keras``,
- foundation models such as `TabICL <https://github.com/soda-inria/tabicl>`_ and
  `TabPFN <https://github.com/PriorLabs/TabPFN>`_.
"""

# %%
# Loading a binary classification dataset
# =======================================
#
# In this example, we tackle the Breast Cancer Winconsin dataset where the goal is a
# binary classification task, i.e. predicting whether a tumor is malignant or benign.
# We use this dataset to keep this example simple.

# %%
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
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
# We shall explore the dataset using `skrub` library's `TableReport`.

# %%
from skrub import TableReport

TableReport(X)

# %%
import pandas as pd

TableReport(pd.DataFrame(y))

# %%
pd.DataFrame(y).value_counts(normalize=True).round(2)

# %%
# Preprocessing
# -------------

# %%
# From the table report, we can make a few inferences:
#
# - The *Stats* tab shows there are no null values.
# - The *Distribution* tab shows us there is moderate level of imbalance: 63% benign from the mean value and 37% malignant values. While we can balance this or add class weights in our neural network, it is important to note that we're modeling the real world
#   and not to achieve an artificial balance!
# - The *Distribution* tab shows a few features that have some outliers, namely: ``radius error``, ``texture error``, ``perimeter error``, ``area error``, ``smoothness error``, ``compactness error``, ``concavity error``, ``concave points error``, ``symmetry error``, ``fractal dimension error``, ``worst area``, ``worst symmetry``, ``worst fractal dimensions``.
#   However, it is important to note that the outliers range from 1-8 in all the different columns which is not huge to cause problems in our modeling.
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

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train_numpy, X_test_numpy, y_train, y_test = train_test_split(
    X_numpy, y, random_state=42
)

# %%
# We can see how `skore` gives us 2 warnings on split to help us with our modeling approach.
#
# - Adhering to the `HighClassImbalanceTooFewExamplesWarning`, we shall employ cross validation in our modeling approach.
# - The `ShuffleTrueWarning` can be ignored as there are no temporal dependencies in our medical dataset. Each example is IID.

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
# that `skore` can utilize, we shall wrap this in `skorch`'s NeuralNetClassifier.

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
# Next, we create our final model by wrapping this into a sklearn `Pipeline` that adds a StandardScaler.

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), nnet)
model

# %%
# We use our X_train and y_train set using some specific set of hyperparameters to evaluate our hyperparameter selection. For the sake of example, we do not test against multiple hyperparameters but a single one.
#
# You may want to look into `ComparisonReport` that can allow you to compare multiple `CrossValidationReport` instances.

# %%
from skore import CrossValidationReport

cv_report = CrossValidationReport(
    estimator=model, X=X_train_numpy, y=y_train, cv_splitter=3
)

# %%
# We can observe the different ROC curves and cumulative scores across different folds for our binary classification task.

# %%
cv_report.metrics.roc().plot()
cv_report.metrics.summarize(
    aggregate=["mean", "std"], indicator_favorability=True
).frame()

# %%
# Once we have picked the best hyperparmeter configuration, we can create a EstimatorReport with a new model object having best hyperparameter settings along with the test sets to get an evaluation report.

# %%
from skore import EstimatorReport

report = EstimatorReport(
    model,
    fit=True,
    X_train=X_train_numpy,
    y_train=y_train,
    X_test=X_test_numpy,
    y_test=y_test,
)

# %%
# Similar to the above, we can observe the report and the ROC curves of our final model.

# %%
report.metrics.roc().plot()
report.metrics.summarize(indicator_favorability=True).frame()

# %%

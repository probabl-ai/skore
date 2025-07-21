"""
.. _example_sklearn_api:

=====================================================================================
Using skore with scikit-learn compatible estimators
=====================================================================================

This example shows how to use skore with scikit-learn compatible estimators.

Any model that can be used with the scikit-learn API can be used with skore.
Skore's :class:`~skore.EstimatorReport` can be used to report on any estimator
that has a ``fit`` and ``predict`` method.
In fact, skore only requires the ``predict`` method if the estimator has already
been fitted.

.. note::

  When computing the ROC AUC or ROC curve for a classification task, the estimator must
  have a ``predict_proba`` method.

In this example, we showcase a gradient boosting model
(`XGBoost <https://github.com/dmlc/xgboost>`_) and a custom estimator.

Note that this example is not exhaustive; many other scikit-learn compatible models can
be used with skore:

-   More gradient boosting libraries like
    `LightGBM <https://github.com/microsoft/LightGBM>`_, and
    `CatBoost <https://github.com/catboost/catboost>`_,

-   Deep learning frameworks such as `Keras <https://github.com/keras-team/keras>`_
    and `skorch <https://github.com/skorch-dev/skorch>`_ (a wrapper for
    `PyTorch <https://github.com/pytorch/pytorch>`_).

-   Tabular foundation models such as
    `TabICL <https://github.com/soda-inria/tabicl>`_ and
    `TabPFN <https://github.com/PriorLabs/TabPFN>`_,

-   etc.

"""

# %%
# Loading a binary classification dataset
# =======================================
#
# We generate a synthetic binary classification dataset with only 1,000 samples to keep
# the computation time reasonable:

# %%
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1_000, random_state=42)
print(f"{X.shape = }")

# %%
# We split our data:

# %%
from skore import train_test_split

split_data = train_test_split(X, y, random_state=42, as_dict=True)

# %%
# Gradient-boosted decision trees with XGBoost
# ============================================
#
# For this binary classification task, we consider a gradient-boosted decision trees
# model from a library external to scikit-learn.
# One of the most popular is `XGBoost <https://github.com/dmlc/xgboost>`_.

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
# We can also inspect our model:

# %%
xgb_report.feature_importance.permutation()

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
import numpy as np


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
#   The estimator above does not have a `predict_proba` method, therefore
#   we cannot display the ROC as above.

# %%
# We can now use this model with skore:

# %%
custom_report = EstimatorReport(CustomClassifier(), pos_label=1, **split_data)
custom_report.metrics.precision()

# %%
# Conclusion
# ==========
#
# This example demonstrates how skore can be used with scikit-learn compatible estimators.
# This allows practitioners to use consistent reporting and visualization tools across different estimators.
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

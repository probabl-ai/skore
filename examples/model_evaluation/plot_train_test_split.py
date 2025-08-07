"""
.. _example_train_test_split:

============================================================
`train_test_split`: get diagnostics when splitting your data
============================================================

This example illustrates the motivation and the use of skore's
:func:`skore.train_test_split` to get assistance when developing ML/DS projects.
"""

# %%
# Train-test split in scikit-learn
# ================================
#
# Scikit-learn has a function for splitting the data into train and test
# sets: :func:`sklearn.model_selection.train_test_split`.
# Its signature is the following:
#
# .. code-block:: python
#
#     sklearn.model_selection.train_test_split(
#         *arrays,
#         test_size=None,
#         train_size=None,
#         random_state=None,
#         shuffle=True,
#         stratify=None
#     )
#
# where ``*arrays`` is a Python ``*args`` (it allows you to pass a varying number of
# positional arguments) and the scikit-learn doc indicates that it is ``a sequence of
# indexables with same length / shape[0]``.

# %%
# Let us construct a design matrix ``X`` and target ``y`` to illustrate our point:

# %%
import numpy as np

X = np.arange(10).reshape((5, 2))
y = np.arange(5)
print(f"{X = }\n{y = }")

# %%
# In scikit-learn, the most common usage is the following:

# %%
from sklearn.model_selection import train_test_split as sklearn_train_test_split

X_train, X_test, y_train, y_test = sklearn_train_test_split(
    X, y, test_size=0.2, random_state=0
)
print(f"{X_train = }\n{y_train = }\n{X_test = }\n{y_test = }")

# %%
# Notice the shuffling that is done by default.

# %%
# In scikit-learn, the user cannot explicitly set the design matrix ``X`` and
# the target ``y``. The following:
#
# .. code-block:: python
#
#   X_train, X_test, y_train, y_test = sklearn_train_test_split(
#       X=X, y=y, test_size=0.2, random_state=0)
#
# would return:
#
# .. code-block:: python
#
#   TypeError: got an unexpected keyword argument 'X'
#
# In general, in Python, keyword arguments are useful to prevent typos. For example,
# in the following, ``X`` and ``y`` are reversed:

# %%
X_train, X_test, y_train, y_test = sklearn_train_test_split(
    y, X, test_size=0.2, random_state=0
)
print(f"{X_train = }\n{y_train = }\n{X_test = }\n{y_test = }")

# %%
# but Python will not catch this mistake for us.
# This is where skore comes in handy.

# %%
# Train-test split in skore
# =========================

# %%
# Skore has its own :func:`skore.train_test_split` that wraps scikit-learn's
# :func:`sklearn.model_selection.train_test_split`.

# %%
X = np.arange(10_000).reshape((5_000, 2))
y = [0] * 2_500 + [1] * 2_500

# %%
# Expliciting the positional arguments for ``X`` and ``y``
# --------------------------------------------------------

# %%
# First of all, naturally, it can be used as a simple drop-in replacement for
# scikit-learn:

# %%
import skore

X_train, X_test, y_train, y_test = skore.train_test_split(
    X, y, test_size=0.2, random_state=0
)

# %%
# .. note::
#
#   The outputs of :func:`skore.train_test_split` are intentionally exactly the same as
#   :func:`sklearn.model_selection.train_test_split`, so the user can just use the
#   skore version as a drop-in replacement of scikit-learn.

# %%
# Contrary to scikit-learn, skore allows users to explicit the ``X`` and ``y``, making
# detection of potential issues easier:

# %%
X_train, X_test, y_train, y_test = skore.train_test_split(
    X=X, y=y, test_size=0.2, random_state=0
)
X_train_explicit = X_train.copy()

# %%
# Moreover, when passing ``X`` and ``y`` explicitly, the ``X``'s are always returned
# before the ``y``'s, even when they are inverted:

# %%
arr = X.copy()
arr_train, arr_test, X_train, X_test, y_train, y_test = skore.train_test_split(
    arr, y=y, X=X, test_size=0.2, random_state=0
)
X_train_explicit_inverted = X_train.copy()

print("When expliciting, with the small typo, are the `X_train`'s still the same?")
print(np.allclose(X_train_explicit, X_train_explicit_inverted))

# %%
# Returning a dictionary instead of positional arguments
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%

from sklearn.linear_model import LogisticRegression
from skore import EstimatorReport

split_data = skore.train_test_split(X=X, y=y, random_state=42, as_dict=True)
estimator = LogisticRegression(random_state=42)
estimator_report = EstimatorReport(estimator, **split_data)


# %%
# Automatic diagnostics: raising methodological warnings
# ------------------------------------------------------
#
# In this section, we show how skore can provide methodological checks.
#
# Class imbalance
# ^^^^^^^^^^^^^^^
#
# In machine learning, class imbalance (the classes in a dataset are not equally
# represented) requires a specific modelling.
# For example, in a dataset with 95% majority class (class ``1``) and 5% minority class
# (class ``0``), a dummy model that always predicts class ``1`` will have a 95%
# accuracy, while it would be useless for identifying examples of class ``0``.
# Hence, it is important to detect when we have class imbalance.
#
# Suppose that we have imbalanced data:

# %%
X = np.arange(10_000).reshape((5_000, 2))
y = [0] * 4_000 + [1] * 1_000

# %%
# In that case, :func:`skore.train_test_split` raises a ``HighClassImbalanceWarning``:

# %%
X_train, X_test, y_train, y_test = skore.train_test_split(
    X=X, y=y, test_size=0.2, random_state=0
)

# %%
# Hence, skore recommends the users to take into account this class imbalance, that
# they might have missed, in their modelling strategy.

# %%
# Moreover, skore also detects class imbalance with a class that has too few samples
# with a ``HighClassImbalanceTooFewExamplesWarning``:

X = np.arange(400).reshape((200, 2))
y = [0] * 150 + [1] * 50

X_train, X_test, y_train, y_test = skore.train_test_split(
    X=X, y=y, test_size=0.2, random_state=0
)

# %%
# Shuffling without a random state
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For `reproducible results across executions
# <https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness>`_,
# skore recommends the use of the ``random_state`` parameter when shuffling
# (remember that ``shuffle=True`` by default) with a ``RandomStateUnsetWarning``:

X = np.arange(10_000).reshape((5_000, 2))
y = [0] * 2_500 + [1] * 2_500

X_train, X_test, y_train, y_test = skore.train_test_split(X=X, y=y, test_size=0.2)

# %%
# Time series data
# ^^^^^^^^^^^^^^^^

# %%
# Now, let us assume that we have `time series data
# <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data>`_:
# the data is somewhat time-ordered:

# %%
import pandas as pd
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X, y = dataset.X, dataset.y
X["date_first_hired"] = pd.to_datetime(X["date_first_hired"], format="%m/%d/%Y")
X.head(2)

# %%
# We can observe that there is a ``date_first_hired`` which is time-based.
#
# As one can not shuffle time (time only moves in one direction: forward), we
# recommend using :class:`sklearn.model_selection.TimeSeriesSplit` instead of
# :func:`sklearn.model_selection.train_test_split` (or :func:`skore.train_test_split`)
# with a ``TimeBasedColumnWarning``:

# %%
X_train, X_test, y_train, y_test = skore.train_test_split(
    X, y, random_state=0, shuffle=False
)


"""
.. _example_train_test_split:

==========================
Enhancing train-test split
==========================

This example illustrates the motivation and the use of skore's
:func:`skore.train_test_split` to get assistance when developing your
ML/DS projects.
"""

# %%
# Creating and loading the skore project
# ======================================
#
# We start by creating a temporary directory to store our project such that we can
# easily clean it after executing this example. If you want to keep the project, you
# have to skip this section.

# %%
import tempfile
from pathlib import Path

import skore

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

my_project = skore.create("my_project.skore", working_dir=temp_dir_path)

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
# where ``*arrays`` is a Python *args (it allows you to pass a varying number of
# positional arguments) and the scikit-learn doc indicates that it is ``a sequence of
# indexables with same length / shape[0]``.

# %%
# Let us construct a design matrix ``X`` and target ``y`` to illustrate our point:

# %%
import numpy as np

X, y = np.arange(10).reshape((5, 2)), range(5)
print(X)
print(y)

# %%
# In scikit-learn, the most common usage is the following:

# %%
from sklearn.model_selection import train_test_split as sklearn_train_test_split

X_train, X_test, y_train, y_test = sklearn_train_test_split(
    X, y, test_size=0.2, random_state=0
)
print(X_train)
print(y_train)

print(X_test)
print(y_test)

# %%
# Notice the shuffling that is done by default.

# %%
# In scikit-learn, the user can not explicitly set the design matrix ``X`` and
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
# In general, in Python, positional arguments are useful to prevent typos such as:

# %%
X_train, X_test, y_train, y_test = sklearn_train_test_split(
    y, X, test_size=0.2, random_state=0
)

# %%
# where ``y`` and ``X`` are inverted in the arguments and no warning is returned.
# This is where skore comes in handy.

# %%
# Train-test split in skore
# =========================

# %%
# skore has its own :func:`skore.train_test_split` that wraps scikit-learn's
# :func:`sklearn.model_selection.train_test_split`.

# %%
# Expliciting the positional arguments for ``X`` and ``y``
# --------------------------------------------------------

# %%
# First of all, naturally, it can be used as a simple drop-in replacement for
# scikit-learn:

# %%
X_train, X_test, y_train, y_test = skore.train_test_split(
    X, y, test_size=0.2, random_state=0
)
print(X_train, y_train)
print(X_test, y_test)

# %%
# .. note::
#
#   The outputs of :func:`skore.train_test_split` are intentionally exactly the same as
#   :func:`sklearn.model_selection.train_test_split`, so the user can just use the
#   skore version as a drop-in replacement of scikit-learn.

# %%
# Contrary to scikit-learn, skore allows users to explicit the``X`` and ``y``, making
# detection of eventual issues easier:

# %%
X_train, X_test, y_train, y_test = skore.train_test_split(
    X=X, y=y, test_size=0.2, random_state=0
)
print(X_train, y_train)
print(X_test, y_test)

# %%
# Moreover, when passing ``X`` and ``y`` explicitly, the ``X``'s are always returned
# before the ``y``'s, even when they are inverted:

# %%
arr = np.arange(10).reshape((5, 2))
arr_train, arr_test, X_train, X_test, y_train, y_test = skore.train_test_split(
    arr, y=y, X=X, test_size=0.2, random_state=0
)
print(X_train, y_train)
print(X_test, y_test)

# %%
# Automatic diagnostics: raising methodological warnings
# ------------------------------------------------------
#
# In machine learning, class-imbalance (the classes in a dataset are not equally
# represented) requires a specific modelling.
# For example, in a dataset with 95% majority class (class ``1``) and 5% minority class
# (class ``0``), a dummy model that always predicts class ``1`` will have a 95%
# accuracy, while it would be useless for identifying examples of class ``0``.
# Hence, it is important to detect when we have class-imbalance.
#
# Suppose that we have imbalanced data :

# %%
X = [[1]] * 4
y = [0, 1, 1, 1]

# %%
# In that case, :func:`skore.train_test_split` raises a warning telling the user that
# there is class imbalance:

# %%
X_train, X_test, y_train, y_test = skore.train_test_split(
    X=X, y=y, test_size=0.2, random_state=0
)

# %%
# Hence, skore recommends the users to take into account this class-imbalance, that
# they might have missed, in their modelling strategy.
# skore provides methodological checks.

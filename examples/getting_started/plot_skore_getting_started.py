"""
.. _example_skore_getting_started:

======================
Skore: getting started
======================
"""

# %%
# This getting started guide illustrates how to use skore and why:
#
# #.    Track your ML/DS results using skore's :class:`~skore.Project`
#       (for storage).
#
# #.    Machine learning diagnostics: get assistance when developing your ML/DS
#       projects to avoid common pitfalls and follow recommended practices.
#
#       * Enhancing key scikit-learn features with :class:`skore.CrossValidationReport`
#         and :func:`skore.train_test_split`.

# %%
# Track and visualize: skore project
# ==================================
#
# A key feature of skore is its :class:`~skore.Project` that allows to store
# items of many types.

# %%
# Setup: creating and loading a skore project
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# .. note::
#
#   If we do not wish for our skore project to be stored in a *temporary* folder, we
#   can simply create and load the project in the current directory with:
#
#   .. code-block:: python
#
#     import skore
#
#     my_project = skore.open("my_project")
#
#   This would create a skore project directory named ``my_project.skore`` in our
#   current directory.

# %%
# Here, we start by creating a temporary directory to store our project so that we can
# easily clean it after executing this example:

# %%
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

# %%
# Then, we create and load the skore project from this temporary directory:

# %%
import skore

my_project = skore.open(temp_dir_path / "my_project")

# %%
# Now that the project exists, we can write some Python code (in the same
# directory) to add (:func:`~skore.Project.put`) some useful items in it,
# with a key-value convention:

# %%
my_project.put("my_int", 3)

# %%
# We can retrieve the value of an item:

# %%
my_project.get("my_int")

# %%
# Skore project: storing some items
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As an illustration of the usage of the skore project with a machine learning
# motivation, let us perform a hyperparameter sweep and store relevant information
# in the skore project.

# %%
# We search for the ``alpha`` hyperparameter of a Ridge regression on the
# Diabetes dataset:

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

X, y = load_diabetes(return_X_y=True)

gs_cv = GridSearchCV(
    Ridge(),
    param_grid={"alpha": np.logspace(-3, 5, 50)},
    scoring="neg_root_mean_squared_error",
)
gs_cv.fit(X, y)

# %%
# Now, we store the hyperparameter's metrics in a dataframe and make a custom
# plot:

# %%
import pandas as pd

df = pd.DataFrame(gs_cv.cv_results_)
df.insert(len(df.columns), "rmse", -df["mean_test_score"].values)
df[["param_alpha", "rmse"]].head()

# %%
import matplotlib.pyplot as plt

fig = plt.figure(layout="constrained")
plt.plot(df["param_alpha"], df["rmse"])
plt.xscale("log")
plt.xlabel("Alpha hyperparameter")
plt.ylabel("RMSE")
plt.title("Ridge regression")
plt.show()

# %%
#
# Finally, we store these relevant items in our skore project, so that we
# can visualize them later:

# %%
my_project.put("my_gs_cv", gs_cv)
my_project.put("my_df", df)
my_project.put("my_fig", fig)

# %%
# .. seealso::
#
#   For more information about the functionalities and the different types
#   of items that we can store in a skore :class:`~skore.Project`,
#   see :ref:`example_working_with_projects`.

# %%
# Tracking the history of items
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Suppose we store several integer values for a same item called ``my_int`:
#
# .. code-block:: python
#
#     import time
#
#     my_project.put("my_int", 4)
#
#     my_project.put("my_int", 9)
#
#     my_project.put("my_int", 16)
#
# Skore does not overwrite items with the same name (key value), instead it stores
# their history so that nothing is lost.
#
# These tracking functionalities are very useful to:
#
# * never lose some key machine learning metrics,
# * and observe the evolution over time / runs.
#
# .. seealso::
#
#   For more information about the tracking of items using their history,
#   see :ref:`example_tracking_items`.

# %%
# Machine learning diagnostics and evaluation
# ===========================================
#
# Skore re-implements or wraps some key scikit-learn class / functions to automatically
# provide diagnostics and checks when using them, as a way to facilitate good practices
# and avoid common pitfalls.

# %%
# Model evaluation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In order to assist its users when programming, skore has implemented a
# :class:`skore.EstimatorReport` class.
#
# Let us load some synthetic data and get the estimator report for a
# :class:`~sklearn.linear_model.LogisticRegression`:

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skore import EstimatorReport

X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = LogisticRegression()

est_report = EstimatorReport(
    clf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
# Now, we can display the help tree to see all the insights that are available to us
# given that we are doing binary classification:

# %%
est_report.help()

# %%
# We can get the report metrics that was computed for us:

# %%
df_est_report_metrics = est_report.metrics.report_metrics()
df_est_report_metrics

# %%
# We can also plot the ROC curve that was generated for us:

# %%
import matplotlib.pyplot as plt

roc_plot = est_report.metrics.plot.roc()
roc_plot
plt.tight_layout()

# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.EstimatorReport`, see :ref:`example_estimator_report`.


# %%
# Cross-validation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# skore has also implemented a :class:`skore.CrossValidationReport` class that contains
# several :class:`skore.EstimatorReport` for each fold.

# %%
from skore import CrossValidationReport

cv_report = CrossValidationReport(clf, X, y, cv_splitter=5)

# %%
# We display the cross-validation report helper:

# %%
cv_report.help()

# %%
# We display the metrics for each fold:

# %%
df_cv_report_metrics = cv_report.metrics.report_metrics()
df_cv_report_metrics

# %%
# We display the ROC curves for each fold:

# %%
roc_plot = cv_report.metrics.plot.roc()
roc_plot
plt.tight_layout()

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.CrossValidationReport`, see :ref:`example_cross_validate`.

# %%
# Train-test split with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Skore has implemented a :func:`skore.train_test_split` function that wraps
# scikit-learn's :func:`sklearn.model_selection.train_test_split`.
#
# For example, it can raise warnings when there is class imbalance in the data to
# provide methodological advice:

# %%
X = np.arange(400).reshape((200, 2))
y = [0] * 150 + [1] * 50

X_train, X_test, y_train, y_test = skore.train_test_split(
    X=X, y=y, test_size=0.2, random_state=0
)

# %%
# In particular, there is a ``HighClassImbalanceWarning``.
#
# Now, let us load a dataset containing some time series data:

# %%
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X, y = dataset.X, dataset.y
X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
X.head(2)

# %%
# We can observe that there is a ``date_first_hired`` which is time-based.
# Now, let us apply :func:`skore.train_test_split` on this data:

# %%
X_train, X_test, y_train, y_test = skore.train_test_split(
    X, y, random_state=0, shuffle=False
)

# %%
# We get a ``TimeBasedColumnWarning`` advising us to use
# :class:`sklearn.model_selection.TimeSeriesSplit` instead!
# Indeed, we should not shuffle time-ordered data!


# %%
# .. seealso::
#
#   More methodological advice is available.
#   For more information about the motivation and usage of
#   :func:`skore.train_test_split`, see :ref:`example_train_test_split`.

# %%
# .. admonition:: Stay tuned!
#
#   These are only the initial features: skore is a work in progress and aims to be
#   an end-to-end library for data scientists.
#
#   Feedbacks are welcome: please feel free to join our
#   `Discord <http://discord.probabl.ai>`_ or
#   `create an issue <https://github.com/probabl-ai/skore/issues>`_.

# %%
# Cleanup the project
# -------------------
#
# Removing the temporary directory:

# %%
temp_dir.cleanup()

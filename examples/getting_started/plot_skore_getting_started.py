"""
.. _example_skore_getting_started:

======================
Skore: getting started
======================
"""

# %%
# This getting started guide illustrates how to use skore and why:
#
# #.    Get assistance when developing your ML/DS projects to avoid common pitfalls
#       and follow recommended practices.
#
#       *   :class:`skore.EstimatorReport`: get an insightful report on your estimator,
#           for evaluation and inspection
#
#       *   :class:`skore.CrossValidationReport`: get an insightful report on your
#           cross-validation results
#
#       *   :class:`skore.ComparisonReport`: benchmark your skore estimator reports
#
#       *   :func:`skore.train_test_split`: get diagnostics when splitting your data
#
# #.    Track your ML/DS results using skore's :class:`~skore.Project`
#       (for storage).

# %%
# Machine learning evaluation and diagnostics
# ===========================================
#
# Skore implements new tools or wraps some key scikit-learn class / functions to
# automatically provide insights and diagnostics when using them, as a way to
# facilitate good practices and avoid common pitfalls.

# %%
# Model evaluation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In order to assist its users when programming, skore has implemented a
# :class:`skore.EstimatorReport` class.
#
# Let us load a binary classification dataset and get the estimator report for a
# :class:`~sklearn.ensemble.RandomForestClassifier`:

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from skore import EstimatorReport

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf_report = EstimatorReport(
    rf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
# Now, we can display the helper to see all the insights that are available to us
# (skore detected that we are doing binary classification):

# %%
rf_report.help()

# %%
# We can get the report metrics that was computed for us:

# %%
rf_report_metrics = rf_report.metrics.report_metrics(pos_label=1)
rf_report_metrics

# %%
# We can also plot the ROC curve that was generated for us:

# %%
import matplotlib.pyplot as plt

roc_plot = rf_report.metrics.roc()
roc_plot.plot()
plt.tight_layout()

# %%
# Furthermore, we can inspect the model using the permutation feature importance:

# %%
rf_report.feature_importance.feature_permutation().T.boxplot(vert=False)
plt.tight_layout()

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.EstimatorReport`, see :ref:`example_estimator_report` for evaluation
#   and :ref:`example_feature_importance` for inspection.


# %%
# Cross-validation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# skore has also (re-)implemented a :class:`skore.CrossValidationReport` class that
# contains several :class:`skore.EstimatorReport` for each fold.

# %%
from skore import CrossValidationReport

cv_report = CrossValidationReport(rf, X, y, cv_splitter=5)

# %%
# We display the cross-validation report helper:

# %%
cv_report.help()

# %%
# We display the metrics for each fold:

# %%
df_cv_report_metrics = cv_report.metrics.report_metrics(pos_label=1)
df_cv_report_metrics

# %%
# We display the ROC curves for each fold:

# %%
roc_plot_cv = cv_report.metrics.roc()
roc_plot_cv.plot()
plt.tight_layout()

# %%
# We can retrieve the estimator report of a specific fold to investigate further,
# for example the first fold:

# %%
report_fold = cv_report.estimator_reports_[0]
report_fold_metrics = report_fold.metrics.report_metrics(pos_label=1)
report_fold_metrics

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.CrossValidationReport`, see :ref:`example_use_case_employee_salaries`.

# %%
# Comparing estimators reports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`skore.ComparisonReport` enables users to compare several estimator reports
# (corresponding to several estimators) on a same test set, as in a benchmark of
# estimators.
#
# Apart from the previous ``rf_report``, let use define another estimator report:

# %%
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
gb_report = EstimatorReport(
    gb, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

gb_report_metrics = gb_report.metrics.report_metrics(pos_label=1)
gb_report_metrics

# %%
# We can conveniently compare our two estimator reports, that were applied to the exact
# same test set:

# %%
from skore import ComparisonReport

comparator = ComparisonReport(reports=[rf_report, gb_report])

# %%
# As for the :class:`~skore.EstimatorReport` and the
# :class:`~skore.CrossValidationReport`, we have a helper:

# %%
comparator.help()

# %%
# Let us display the result of our benchmark:

# %%
benchmark_metrics = comparator.metrics.report_metrics(pos_label=1)
benchmark_metrics

# %%
# We have the result of our benchmark.

# %%
# We display the ROC curve for the two estimator reports we want to compare, by
# superimposing them on the same figure:

# %%
comparator.metrics.roc().plot()
plt.tight_layout()

# %%
# Train-test split with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Skore has implemented a :func:`skore.train_test_split` function that wraps
# scikit-learn's :func:`sklearn.model_selection.train_test_split`.
#
# Let us load a dataset containing some time series data:

# %%
import pandas as pd
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X, y = dataset.X, dataset.y
X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
X.head(2)

# %%
# We can observe that there is a ``date_first_hired`` which is time-based.
# Now, let us apply :func:`skore.train_test_split` on this data:

# %%
import skore

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
# Tracking: skore project
# =======================
#
# Another key feature of skore is its :class:`~skore.Project` that allows to store
# items of many types.

# %%
# Setup: creating and loading a skore project
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
#  Let's start by creating a skore project directory named ``my_project.skore`` in our
#  current directory:

# %%

# sphinx_gallery_start_ignore
import os
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = Path(temp_dir.name)
os.chdir(temp_dir_path)
# sphinx_gallery_end_ignore
my_project = skore.Project("my_project")

# %%
# Skore project: storing and retrieving some items
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that the project exists, we can store some useful items in it (in the same
# directory) using :func:`~skore.Project.put`), with a "universal" key-value convention,
# along with some annotations.

# %%
# Let us store our report metrics and the model report on the random forest using
# :meth:`~skore.Project.put`, along with some annotation to help us track our
# experiments:

# %%
my_project.put(
    "report_metrics", rf_report_metrics, note="random forest, pandas dataframe"
)
my_project.put(
    "estimator_report", rf_report, note="random forest, skore estimator report"
)

# %%
# .. note ::
#   With the skore :func:`~skore.Project.put`, there is no need to remember the API for
#   each type of object: ``df.to_csv(...)``, ``plt.savefig(...)``, ``np.save(...)``,
#   etc.

# %%
# We can retrieve the value of an item using :meth:`~skore.Project.get`:

# %%
my_project.get("report_metrics")

# %%
# We can also retrieve the storage data and annotation:

# %%
my_project.get("report_metrics", metadata="all")

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
# Now, from the gradient boosting model, let us the exact same keys:

# %%
my_project.put(
    "report_metrics", gb_report_metrics, note="gradient boosting, pandas dataframe"
)
my_project.put(
    "estimator_report", gb_report, note="gradient boosting, skore estimator report"
)


# %%
# Skore does not overwrite items with the same name (key): instead, it stores
# their history so that nothing is lost:

# %%
history = my_project.get("report_metrics", version="all")
# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore
history

# %%
# These tracking functionalities are very useful to:
#
# * never lose some key machine learning metrics,
# * and observe the evolution over time / runs.

# %%
# .. seealso::
#
#   For more functionalities about the tracking of items using their history,
#   see :ref:`example_tracking_items`.

# %%
# .. admonition:: Stay tuned!
#
#   These are only the initial features: skore is a work in progress and aims to be
#   an end-to-end library for data scientists.
#
#   Feedbacks are welcome: please feel free to join our
#   `Discord <http://discord.probabl.ai>`_ or
#   `create an issue <https://github.com/probabl-ai/skore/issues>`_.

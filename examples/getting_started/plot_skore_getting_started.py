"""
.. _example_skore_getting_started:

======================
Skore: getting started
======================
"""

# %%
# This getting started guide illustrates how to use skore and why:
#
# #.    Get assistance when developing your machine learning projects to avoid common
#       pitfalls and follow recommended practices.
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
# #.    Track your machine learning results using skore's :class:`~skore.Project`
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
# .. note::
#   This helper is great because:
#
#   -   it enables users to get a glimpse at the API of the different available
#       accessors without having to look up the online documentation,
#   -   it provides methodological guidance: for example, we easily provide
#       several metrics as a way to encourage users looking into them.

# %%
# We can evaluate our model using the :meth:`~skore.EstimatorReport.metrics` accessor.
# In particular, we can get the report metrics that is computed for us (including the
# fit and prediction times):

# %%
rf_report.metrics.report_metrics(pos_label=1)

# %%
# For inspection, we can also retrieve the predictions, on the train set for example
# (here we display only the first 10 predictions for conciseness purposes):

# %%
rf_report.get_predictions(data_source="train")[0:10]

# %%
# We can also plot the ROC curve that is generated for us:

# %%
roc_plot = rf_report.metrics.roc()
roc_plot.plot()

# %%
# Furthermore, we can inspect our model using the
# :meth:`~skore.EstimatorReport.feature_importance` accessor.
# In particular, we can inspect the model using the permutation feature importance:

# %%
import matplotlib.pyplot as plt

rf_report.feature_importance.permutation(seed=0).T.boxplot(vert=False)
plt.tight_layout()

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.EstimatorReport`, see the following use cases:
#
#   -   :ref:`example_estimator_report` for model evaluation,
#   -   :ref:`example_feature_importance` for model inspection.

# %%
# Cross-validation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# skore has also (re-)implemented a :class:`skore.CrossValidationReport` class that
# contains several :class:`skore.EstimatorReport`, one for each fold.

# %%
from skore import CrossValidationReport

cv_report = CrossValidationReport(rf, X, y, cv_splitter=5)

# %%
# We display the cross-validation report helper:

# %%
cv_report.help()

# %%
# We display the mean and standard deviation for each metric:

# %%
cv_report.metrics.report_metrics(pos_label=1)

# %%
# or by individual fold:

# %%
cv_report.metrics.report_metrics(aggregate=None, pos_label=1)

# %%
# We display the ROC curves for each fold:

# %%
roc_plot_cv = cv_report.metrics.roc()
roc_plot_cv.plot()

# %%
# We can retrieve the estimator report of a specific fold to investigate further,
# for example getting the report metrics for the first fold only:

# %%
cv_report.estimator_reports_[0].metrics.report_metrics(pos_label=1)

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.CrossValidationReport`, see :ref:`example_use_case_employee_salaries`.

# %%
# Comparing estimator reports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`skore.ComparisonReport` enables users to compare several estimator reports
# (corresponding to several estimators) on a same test set, as in a benchmark of
# estimators.
#
# Apart from the previous ``rf_report``, let us define another estimator report:

# %%
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
gb_report = EstimatorReport(
    gb, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

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
comparator.metrics.report_metrics(pos_label=1)

# %%
# Thus, we easily have the result of our benchmark for several recommended metrics.

# %%
# Moreover, we can display the ROC curve for the two estimator reports we want to
# compare, by superimposing them on the same figure:

# %%
comparator.metrics.roc().plot()

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

dataset_employee = fetch_employee_salaries()
X_employee, y_employee = dataset_employee.X, dataset_employee.y
X_employee["date_first_hired"] = pd.to_datetime(
    X_employee["date_first_hired"], format="%m/%d/%Y"
)
X_employee.head(2)

# %%
# We can observe that there is a ``date_first_hired`` which is time-based.
# Now, let us apply :func:`skore.train_test_split` on this data:

# %%
import skore

_ = skore.train_test_split(
    X=X_employee, y=y_employee, random_state=0, shuffle=False, as_dict=True
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
# Another key feature of skore is its :class:`~skore.Project` that allows us to store
# and retrieve items of many types.

# %%
# Setup: creating and loading a skore project
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
#  Let us start by creating a skore project directory named ``my_project.skore`` in our
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
# directory) using :func:`~skore.Project.put`, with a "universal" key-value convention,
# along with some annotations.

# %%
# Let us store the accuracy and the estimator report of the random forest using
# :meth:`~skore.Project.put`, along with some annotation to help us track our
# experiments:

# %%
my_project.put("accuracy", rf_report.metrics.accuracy(), note="random forest, float")
my_project.put(
    "estimator_report", rf_report, note="random forest, skore estimator report"
)

# %%
# .. note ::
#   With the skore :func:`~skore.Project.put`, there is no need to remember the API for
#   saving or exporting each type of object: ``df.to_csv(...)``, ``plt.savefig(...)``,
#   ``np.save(...)``, etc.
#   There is also the unified :func:`~skore.Project.get` for loading items.

# %%
# We can retrieve the value of an item using :meth:`~skore.Project.get`:

# %%
my_project.get("accuracy")

# %%
# We can also retrieve the storage date and our annotation:

# %%
from pprint import pprint

accuracies = my_project.get("accuracy", metadata="all")
pprint(accuracies)

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
# Now, for the gradient boosting model, let us store the same kinds of items
# using the exact same keys, namely ``accuracy`` and ``estimator_report``:

# %%
my_project.put(
    "accuracy", gb_report.metrics.accuracy(), note="gradient boosting, float"
)
my_project.put(
    "estimator_report", gb_report, note="gradient boosting, skore estimator report"
)

# %%
# Skore does not overwrite items with the same name (key): instead, it stores
# their history so that nothing is lost:

# %%
accuracies_history = my_project.get("accuracy", metadata="all", version="all")
# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore
pprint(accuracies_history)

# %%
# .. note::
#   These tracking functionalities are very useful to:
#
#   -   never lose some key machine learning metrics,
#   -   and observe the evolution over time / runs.

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

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
# Skore implements new tools or wraps some key scikit-learn concepts to
# automatically provide insights and diagnostics when using them, as a way to
# accelerate development and facilitate good practices.

# %%
# Model evaluation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# One of the main skore features is the :class:`skore.EstimatorReport` class.
#
# Let us create a challenging synthetic binary classification dataset and train a
# :class:`~sklearn.ensemble.RandomForestClassifier` on it, then wrap an
# :class:`~skore.EstimatorReport` around it.

# %%
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skore import EstimatorReport

X, y = make_classification(
    n_samples=10_000,
    n_classes=3,
    class_sep=0.3,
    n_clusters_per_class=1,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rf = RandomForestClassifier(random_state=0)

rf_report = EstimatorReport(
    rf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
# Now, we can display the help menu to see what we can do with the report:

# %%
rf_report.help()

# %%
# .. note::
#   This helper:
#
#   -   enables users to get a glimpse at the API of the different available
#       accessors without having to look up the online documentation,
#   -   provides methodological guidance: for example, we easily provide
#       several metrics as a way to encourage users looking into them.
#       In particular, skore automatically detected that we are doing multiclass
#       classification.
#

# %%
# We can use the :meth:`~skore.EstimatorReport.metrics` attribute to evaluate the
# performance of our estimator:

# %%
rf_report.metrics.summarize(indicator_favorability=True).frame()

# %%
# We can also retrieve the predictions directly e.g. on the train set:

# %%
rf_report.get_predictions(data_source="train")[0:10]

# %%
# More metrics are available, such as the ROC curve:

# %%
roc = rf_report.metrics.roc()
roc.plot()

# %%
# We can further inspect our model using the
# :meth:`~skore.EstimatorReport.feature_importance` attribute.
# For example we can compute the permutation feature importance:

# %%
rf_report.feature_importance.permutation(seed=0).T.boxplot(vert=False)

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
# skore can perform cross-validation through the :class:`skore.CrossValidationReport`
# class, which is simply an ensemble of :class:`EstimatorReports <skore.EstimatorReport>` (one for each split).

# %%
from skore import CrossValidationReport

cv_report = CrossValidationReport(rf, X, y, splitter=5)

# %%
# The cross-validation report also has a helper menu:

# %%
cv_report.help()

# %%
# It can also compute common metrics:

# %%
cv_report.metrics.summarize().frame()

# %%
# If aggregation is not necessary:

# %%
cv_report.metrics.summarize(aggregate=None).frame()

# %%
# Plot metrics like the ROC curve are also implemented:

# %%
roc_cv = cv_report.metrics.roc()
roc_cv.plot()

# %%
# The cross-validation report contains the individual estimator reports
# for each split, so e.g. we can obtain metrics for the first split only:

# %%
first_split_report = cv_report.estimator_reports_[0]
first_split_report.metrics.summarize().frame()

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.CrossValidationReport`, see :ref:`example_use_case_employee_salaries`.

# %%
# Comparing estimator reports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# skore makes it easy to compare estimators on the same test set through the
# :class:`skore.ComparisonReport` class.
#
# We previously trained a random forest estimator which we investigated in
# ``rf_report``.
# Let us now try another algorithm:

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

gb_report = EstimatorReport(
    HistGradientBoostingClassifier(random_state=0),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    pos_label=1,
)

# %%
# Now let us create a :class:`~skore.ComparisonReport`:

# %%
from skore import ComparisonReport

comparator = ComparisonReport(reports=[rf_report, gb_report])

# %%
# The comparison report also has a helper menu:

# %%
comparator.help()

# %%
# It can also compute common metrics, which will give us our benchmark:

# %%
comparator.metrics.summarize(indicator_favorability=True).frame()

# %%
# Other metrics like the ROC curve are also implemented to enable easy comparison:

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
# The dataset contains a ``date_first_hired`` column which is time-based.
# Now, let us apply :func:`skore.train_test_split` on this data:

# %%
import skore

_ = skore.train_test_split(
    X=X_employee, y=y_employee, random_state=0, shuffle=False, as_dict=True
)

# %%
# We get a ``TimeBasedColumnWarning`` advising us to use
# :class:`sklearn.model_selection.TimeSeriesSplit` instead!
# Indeed, shuffling time-ordered data is a common pitfall.

# %%
# .. seealso::
#
#   More methodological advice is available.
#   For more information about the motivation and usage of
#   :func:`skore.train_test_split`, see :ref:`example_train_test_split`.

# %%
# Tracking work with skore
# ========================
#
# Another key feature of skore is :class:`skore.Project` that allows us to store
# and retrieve :class:`~skore.EstimatorReport` objects.

# %%
# Let us create a skore project:

# %%

# sphinx_gallery_start_ignore
import os
import tempfile

temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["SKORE_WORKSPACE"] = temp_dir.name
# sphinx_gallery_end_ignore
my_project = skore.Project("my_project")

# %%
# Storing reports
# ^^^^^^^^^^^^^^^
#
# Now that the project exists, we can store reports in it using
# :func:`~skore.Project.put`, with a key-value convention.

# %%
# Let us store the estimator reports of the random forest and the gradient boosting
# models to keep track of our experiments:

# %%
my_project.put("my_estimator_report", rf_report)
my_project.put("my_estimator_report", gb_report)

# %%
# Retrieving stored reports
# ^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# We can retrieve the data in the project in the form of a
# :class:`~skore.project.summary.Summary` object:

# %%
summary = my_project.summarize()
print(type(summary))

# %%
# From there, we can retrieve the complete list of stored reports:

# %%
from pprint import pprint

stored_reports = summary.reports()
pprint(stored_reports)

# %%
# And we can manipulate them as usual, e.g. compare them:

# %%
comparator = ComparisonReport(reports=stored_reports)
comparator.metrics.summarize(pos_label=1, indicator_favorability=True).frame()

# %%
# We can retrieve metrics about our stored estimator reports, for example
# the fit and test times for the first estimator report:

# %%
stored_reports[0].metrics.timings()

# %%
# But what if instead of having stored only 2 estimators reports, we had a dozen or
# even a few hundreds over several months of experimenting?
# We would need a more powerful way to navigate through our stored estimator reports.

# %%
# Searching through project reports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Using the interactive widget
# """"""""""""""""""""""""""""
#
# If rendered in a Jupyter notebook, ``summary`` would render an interactive
# parallel coordinate plot to search for your preferred model based on some metrics,
# which looks like this:
#
# .. image:: /_static/images/screenshot_getting_started.png
#   :alt: Screenshot of the widget in a Jupyter notebook
#
# Each line is an estimator. We can select interesting estimators by
# clicking on the plot: clicking on a line selects one estimator, and dragging on
# one of the metrics axes will trigger a brush tool to select several estimators.
#
# Once we are done selecting on the plot, we can retrieve the selected reports:
#
# .. code:: python
#
#     summary.reports()

# %%
# Using the Python API
# """"""""""""""""""""
#
# Alternatively, we can query for reports using Python (more precisely, the pandas
# query language).

# %%
# We can use the following keys to write queries:

# %%
summary.keys()

# %%
# For example, we can query for all the instances of
# :class:`~sklearn.ensemble.RandomForestClassifier`:

# %%
summary.query("learner.str.contains('RandomForestClassifier')").reports()

# %%
# Or, we can query for all the estimators made for classification:

# %%
pprint(summary.query("ml_task.str.contains('classification')").reports())

# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore

# %%
# .. admonition:: Stay tuned!
#
#   This is only the beginning for skore. We welcome your feedback and ideas
#   to make it the best tool for end-to-end data science.
#
#   Feel free to join our community on `Discord <http://discord.probabl.ai>`_
#   or `create an issue <https://github.com/probabl-ai/skore/issues>`_.

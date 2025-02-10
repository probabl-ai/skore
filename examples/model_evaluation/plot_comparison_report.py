"""
.. _example_compare_estimators:

=====================================================
`ComparisonReport`: Compare several estimator reports
=====================================================

In :ref:`example_estimator_report`, we explain how to use the
:class:`skore.EstimatorReport`.
Let us assume that, for a same dataset, we have several estimator reports corresponding
to several estimators.
We would like to compare these estimator reports together, as in a benchmark of
estimators.
That is what skore implemented a :class:`~skore.ComparisonReport` for.
"""

# %%
# Loading some data and defining 2 estimators
# ===========================================
#
# Let us load and split some data:

# %%
from sklearn.datasets import make_classification
from skore import train_test_split

X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
# We get our first estimator report for a logistic regression:

# %%
from sklearn.linear_model import LogisticRegression
from skore import EstimatorReport

logistic_regression = LogisticRegression(random_state=0)
logistic_regression_report = EstimatorReport(
    logistic_regression,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# %%
# Let us create our second model, a random forest, *on the same data as
# previously*:

# %%
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(max_depth=2, random_state=0)
random_forest_report = EstimatorReport(
    random_forest,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# %%
# Comparing 2 estimator reports
# =============================
#
# Now, we have two estimator reports corresponding to a logistic regression and a
# random forest *on the same dataset*.
# Hence, we can compare these two estimator reports using a
# :class:`~skore.ComparisonReport`:

# %%
from skore import ComparisonReport

comp = ComparisonReport(
    reports=[
        logistic_regression_report,
        random_forest_report,
    ]
)

# %%
# As for the :class:`~skore.EstimatorReport` and the
# :class:`~skore.CrossValidationReport`, we have a helper:

# %%
comp.help()

# %%
# Metrics
# ^^^^^^^

# %%
# Let us display the result of our benchmark:

# %%
df_report_metrics = comp.metrics.report_metrics()
df_report_metrics

# %%
# We can use some pandas styling:


# %%
def highlight_max(s):
    return ["background-color: #ffffb3 ; color:red" if v == s.max() else "" for v in s]


df_report_metrics.style.apply(highlight_max, axis=0).format(precision=3)

# %%
# .. note::
#
#   The highlight indicates the maximum value of each metric: use the (↗︎) or (↘︎) arrows
#   for interpretation.

# %%
# Using the pandas API, we can also plot the accuracy for example:

# %%
import matplotlib.pyplot as plt

comp.metrics.accuracy().plot.barh()
plt.tight_layout()


# %%
# Plots
# ^^^^^

# %%
# We can also compare the ROC curves of all estimators for example:

# %%
import matplotlib.pyplot as plt

display = comp.metrics.roc().plot()
plt.tight_layout()

# %%
# .. note::
#
#   In a near future version of skore, the legends will be correct with the names of
#   the estimators.

# %%
# Sanity checks
# =============
#
# Comparing 2 estimator reports from different datasets
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Methodologically speaking, it makes no sense to compare the metrics of 2 estimators
# that were trained / tested on different datasets: as scores are relative, one can
# only compare what is comparable.
#
# Let us check what happens if we use the :class:`~skore.ComparisonReport` on 2
# estimators reports that were applied to different datasets.

# %%
# Let us load and split a first dataset on which we apply a logistic regression:

# %%
X1, y1 = make_classification(
    n_classes=2, n_samples=100_000, n_informative=4, random_state=0
)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=42)

logistic_regression_report = EstimatorReport(
    logistic_regression,
    X_train=X1_train,
    y_train=y1_train,
    X_test=X1_test,
    y_test=y1_test,
)

# %%
# Let us load and split a second dataset (notice than the ``random_state`` parameter
# changes in :func:`~sklearn.datasets.make_classification`) on which we apply a random
# forest:

# %%
X2, y2 = make_classification(
    n_classes=2, n_samples=100_000, n_informative=4, random_state=1
)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=42)

random_forest_report = EstimatorReport(
    random_forest,
    X_train=X2_train,
    y_train=y2_train,
    X_test=X2_test,
    y_test=y2_test,
)

# %%
# Now, let us try to compare these estimator reports:

# %%
try:
    # We expect skore to raise an error
    comp_datasets = ComparisonReport(
        reports=[
            logistic_regression_report,
            random_forest_report,
        ]
    )
except ValueError as e:
    print(f"skore raised a ValueError with the following message:\n\n{e}")

# %%
# As desired, skore raises an error.

# %%
# Comparing 2 estimator reports that do not hold their data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Methodologically, we always want to compare estimator reports on the exact same
# dataset. If we do not know where the datasets come from, then we expect skore to
# throw a warning:

# %%
logistic_regression_report = EstimatorReport(logistic_regression, fit=False)

random_forest_report = EstimatorReport(random_forest, fit=False)

comp = ComparisonReport(
    reports=[
        logistic_regression_report,
        random_forest_report,
    ]
)

# %%
# As desired, skore raised a warning.

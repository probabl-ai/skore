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
# Let us load and split some data:

# %%
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from skore import train_test_split

X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
# We get our first estimator report for a logistic regression:

# %%
from sklearn.linear_model import LogisticRegression
from skore import EstimatorReport

logistic_regression = LogisticRegression()
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
random_forest = RandomForestClassifier(max_depth=2, random_state=0)
random_forest_report = EstimatorReport(
    random_forest,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# %%
# Now, we have two estimator reports corresponding to a logistic regression and a
# random forest *on the same dataset*.
# Hence, we can compare these two estimator report using a
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
# As for the :class:`~skore.EstimatorReport` and the :class:`~skore.CrossValidationReport`, we have a helper:

# %%
comp.help()

# %%
# Let us display the result of our benchmark:

# %%
df_report_metrics = comp.metrics.report_metrics()
df_report_metrics

# %%
# We can also compare the ROC curves of all estimators for example:

# %%
import matplotlib.pyplot as plt

comp.metrics.plot.roc()
plt.tight_layout()

# %%
# Using pandas, we can also plot accuracy using pandas API
comp.metrics.accuracy().plot.bar()
plt.show()

# %%


def highlight_max(s):
    return ["font-weight: bold" if v == s.max() else "" for v in s]


df_report_metrics.style.apply(highlight_max, axis=0)


# %%
#
#   .. code-block:: python
#
#       comp.metrics.metric()
#       comp.metrics.custom_metric(scoring=my_func)
#

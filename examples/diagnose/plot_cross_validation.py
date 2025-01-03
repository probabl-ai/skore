"""
=======================
Cross-validation report
=======================

This example shows how the :class:`skore.CrossValidationReport` class can be used to
get insights from any scikit-learn estimator.
"""

# %%
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
df = dataset.X
y = dataset.y

# %%
from skrub import TableReport

TableReport(df)

# %%
TableReport(y.to_frame())

# %%
from skrub import tabular_learner

estimator = tabular_learner("regressor")
estimator

# %%
from skore import CrossValidationReport
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
reporter = CrossValidationReport(estimator, df, y, cv=cv, n_jobs=4)
reporter

# %%
results = reporter.metrics.r2()
results

# %%
results.aggregate(["mean", "std"], axis=0)

# %%
ax = results.plot.kde()
ax.set_xlim(0, 1)
_ = ax.set_title("R2 score distribution")

# %%
results = reporter.metrics.report_metrics()
results

# %%
results.aggregate(["mean", "std"], axis=0)

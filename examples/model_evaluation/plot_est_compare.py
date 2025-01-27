"""
.. _example_compare_estimators:

===========
Example driven development
===========
"""

# %%
# From your Python code, create and load a skore :class:`~skore.Project`:

# %%
import skore

# sphinx_gallery_start_ignore
import os
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = Path(temp_dir.name)
os.chdir(temp_dir_path)
# sphinx_gallery_end_ignore
my_project = skore.open("my_project", create=True)

# %%
# This will create a skore project directory named ``my_project.skore`` in your
# current working directory.

# %%
# Evaluate your model using skore's :class:`~skore.CrossValidationReport`:

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from skore import EstimatorReport, train_test_split

X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
log_reg = LogisticRegression()
est_report_lr = EstimatorReport(
    log_reg, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# %%
rf = RandomForestClassifier(max_depth=2, random_state=0)
est_report_rf = EstimatorReport(
    rf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)


# %%
# here starts the example to compare.
# to create the comparison object:
# comp = skore.Comparator(estimator_reports = [est_report_lr, est_report_rf])

# %%
import plotly.express as px

ord = [est.metrics.accuracy().values[0][0] for est in [est_report_lr, est_report_rf]]
abs = [est.metrics.accuracy().index[0] for est in [est_report_lr, est_report_rf]]
fig = px.scatter(x=abs, y=ord)
fig.update_traces(marker_size=10)

# the comparator object has a structure quite similar to the estimators it's comparing.
# comp.metrics.plot(metric = "recall")
fig.show()
# with proper legend of course.
# border cases:
# - what if several estimators have the same model name?


# %%
# for a further iteration
# comp.metrics.plot(metric = "recall", with_time = True) # with_time is False by default
# idea: we would have the training time in abscissa, and the metric in ordinates.
# cf for instance the second plot of this reddit post: https://www.reddit.com/r/LocalLLaMA/comments/191u1j9/visualising_llm_training_compute_correlating_to/#lightbox

# %%
import pandas as pd


def highlight_max(s):
    # thanks chatgpt
    return ["font-weight: bold" if v == s.max() else "" for v in s]


df_metrics = pd.concat(
    [est_report_lr.metrics.report_metrics(), est_report_rf.metrics.report_metrics()],
    ignore_index=False,
)
# border cases:
# - what happen if we have computed a score for one model, and not for the other?
# - what if several estimators have the same model name?

# comp.metrics.metric()
df_metrics.style.apply(highlight_max, axis=0)

# %%
print("fin de l'example")
# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore

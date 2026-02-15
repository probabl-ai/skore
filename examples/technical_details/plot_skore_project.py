"""
.. _example_skore_project:

===================
Local skore Project
===================

This example shows how to use :class:`~skore.Project` in **local** mode: store
reports on your machine and inspect them. A key point is that
:meth:`~skore.Project.summarize` returns a :class:`~skore.project._summary.Summary`,
which is a :class:`pandas.DataFrame`. In Jupyter you get an interactive widget, but
you can always inspect and filter the summary as a DataFrame if you prefer.
"""

# %%
# Create a local project and store reports
# =========================================
#
# We use a temporary directory as the workspace so the example is self-contained.
# In practice you can omit ``workspace`` to use the default (e.g. a ``skore/``
# directory in your user cache).
from pathlib import Path
from tempfile import TemporaryDirectory

from skore import Project

tmpdir = Path(TemporaryDirectory().name)
project = Project("example-project", workspace=tmpdir)

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from skore import train_test_split
from skrub import tabular_pipeline

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
estimator = tabular_pipeline(LogisticRegression(max_iter=1_000))

# %%
import numpy as np
from sklearn.base import clone
from skore import EstimatorReport

for regularization in np.logspace(-7, 7, 31):
    report = EstimatorReport(
        clone(estimator).set_params(logisticregression__C=regularization),
        **split_data,
        pos_label=1,
    )
    project.put(f"lr-regularization-{regularization:.1e}", report)

# %%
# Summarize: you get a DataFrame
# ==============================
#
# :meth:`~skore.Project.summarize` returns a :class:`~skore.project._summary.Summary`,
# which subclasses :class:`pandas.DataFrame`. In a Jupyter environment it renders
# an interactive parallel-coordinates widget by default.
summary = project.summarize()

# %%
# To see the normal DataFrame table instead of the widget (e.g. in scripts or
# when you prefer the table), wrap the summary in :class:`pandas.DataFrame`:
import pandas as pd

pandas_summary = pd.DataFrame(summary)
pandas_summary

# %%
# Basically, our summary contains metadata related to various information that we need
# to quickly help filtering the reports.
summary.info()

# %%
# Filter reports by metric (e.g. keep only those above a given accuracy) and
# work with the result as a table.
summary.query("log_loss < 0.1")["key"].tolist()

# %%
# Use :meth:`~skore.project._summary.Summary.reports` to load the corresponding
# reports from the project (optionally after filtering the summary).
reports = summary.query("log_loss < 0.1").reports(return_as="comparison")
len(reports.reports_)

# %%
# Since we got a :class:`~skore.ComparisonReport`, we can use the metrics accessor
# to summarize the metrics across the reports.
reports.metrics.summarize().frame()

# %%
reports.metrics.roc().plot(subplot_by=None)

# %%
project.delete("example-project", workspace=tmpdir)

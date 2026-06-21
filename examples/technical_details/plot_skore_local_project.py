"""
.. _example_skore_local_project:

===================
Local skore Project
===================

This example shows how to use :class:`~skore.Project` in **local** mode: store
reports on your machine and inspect them. A key point is that
:meth:`~skore.Project.summarize` returns a ``Summary`` object that holds the
metadata and metrics of every report. In Jupyter it renders as an interactive
table with three views (Table, parallel-coordinates Plot, and Trend) where you
can filter and pick reports to build a query string; the underlying
:class:`pandas.DataFrame` is accessible through its ``frame`` method.
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

tmp_dir = TemporaryDirectory()
tmp_path = Path(tmp_dir.name)
project = Project(name="example-project", mode="local", workspace=tmp_path)

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
estimator = tabular_pipeline(LogisticRegression(max_iter=1_000))

# %%
import numpy as np
from sklearn.base import clone
from skore import evaluate

for regularization in np.logspace(-7, 7, 31):
    report = evaluate(
        clone(estimator).set_params(logisticregression__C=regularization),
        X,
        y,
        splitter=0.2,
        pos_label=1,
    )
    project.put(f"lr-regularization-{regularization:.1e}", report)

# %%
# Summarize: you get a Summary
# ============================
#
# :meth:`~skore.Project.summarize` returns a :class:`~skore.Summary` object. In a
# Jupyter environment it renders as an interactive table where you can filter rows and
# pick reports across the different views; the selection produces a query string ready
# to pass to :meth:`~skore.Summary.query`.
summary = project.summarize()
summary

# %%
# Filter reports by metric (e.g. keep only those above a given accuracy) and
# work with the result as a table.
summary.query("log_loss < 0.1")

# %%
# Use :meth:`~skore.Summary.compare` to load the corresponding reports from the
# project (optionally after filtering the summary). Passing ``return_as="report"``
# returns a :class:`~skore.ComparisonReport` built from those reports.
reports = summary.query("log_loss < 0.1").compare(return_as="report")
reports

# %%
_ = reports.metrics.roc().plot(subplot_by=None)

# %%
project.delete("example-project", workspace=tmp_path)
tmp_dir.cleanup()

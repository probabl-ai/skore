"""
.. _example_skore_hub_project:

=================
Hub skore Project
=================

This example shows how to use :class:`~skore.Project` in **hub** mode: store
reports remotely and inspect them. A key point is that
:meth:`~skore.Project.summarize` returns a :class:`~skore.project._summary.Summary`,
which is a :class:`pandas.DataFrame`. In Jupyter you get an interactive widget, but
you can always inspect and filter the summary as a DataFrame if you prefer.

Examples
--------

Basic usage:

.. code-block:: bash

    WORKSPACE=<workspace> PROJECT=<project> python plot_skore_hub_project.py
"""

# %%
# .. testsetup::
from logging import getLogger
from os import environ
from sys import exit

logger = getLogger(__name__)

if environ.get("SPHINX_BUILD"):
    GITHUB = environ.get("GITHUB_ACTIONS")
    API_KEY = environ.get("SPHINX_EXAMPLE_API_KEY")
    WORKSPACE = environ.get("SPHINX_EXAMPLE_WORKSPACE")
    PROJECT = environ.get("SPHINX_EXAMPLE_PROJECT")

    if not (GITHUB and API_KEY and WORKSPACE and PROJECT):
        logger.warning("Example `_example_skore_hub_project` skipped.")
        exit(0)

    environ["SKORE_HUB_API_KEY"] = API_KEY
else:
    assert (WORKSPACE := environ.get("WORKSPACE")), "`WORKSPACE` must be defined."
    assert (PROJECT := environ.get("PROJECT")), "`PROJECT` must be defined."

# %%
from skore import login

login(mode="hub")

# %%
# .. testsetup::
from httpx import HTTPStatusError, codes
from skore import Project

try:
    Project.delete(f"{WORKSPACE}/{PROJECT}", mode="hub")
except HTTPStatusError as e:
    if e.response.status_code != codes.NOT_FOUND:
        raise

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from skore import train_test_split
from skrub import tabular_pipeline

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
estimator = tabular_pipeline(LogisticRegression(max_iter=1_000))

# %%
from numpy import logspace
from sklearn.base import clone
from skore import EstimatorReport, Project

project = Project(f"{WORKSPACE}/{PROJECT}", mode="hub")

for regularization in logspace(-7, 7, 31):
    project.put(
        f"lr-regularization-{regularization:.1e}",
        EstimatorReport(
            clone(estimator).set_params(logisticregression__C=regularization),
            **split_data,
            pos_label=1,
        ),
    )

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

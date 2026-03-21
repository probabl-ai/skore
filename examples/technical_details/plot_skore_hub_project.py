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

To run this example and push in your own Skore Hub workspace and project, you can run
this example with the following command:

.. code-block:: bash

    WORKSPACE=<workspace> PROJECT=<project> python plot_skore_hub_project.py

In this gallery, we are going to push the different reports into a public
workspace.
"""

# sphinx_gallery_start_ignore
#
# Configure the context variables and ensure that the example is run with sufficient
# credentials. This is a useful consistency check for CI where you can't have
# interactive login.
import os

if os.environ.get("SPHINX_BUILD"):
    GITHUB = os.environ.get("GITHUB_ACTIONS")
    API_KEY = os.environ.get("SPHINX_EXAMPLE_API_KEY")
    WORKSPACE = os.environ.get("SPHINX_EXAMPLE_WORKSPACE")
    VERSION = os.environ.get("SPHINX_VERSION")

    if not (GITHUB and API_KEY and WORKSPACE and VERSION):
        raise RuntimeError("Required environment variables not set.")

    PROJECT = f"example-skore-hub-project-{VERSION}"
    os.environ["SKORE_HUB_API_KEY"] = API_KEY
else:
    assert (WORKSPACE := os.environ.get("WORKSPACE")), "`WORKSPACE` must be defined."
    assert (PROJECT := os.environ.get("PROJECT")), "`PROJECT` must be defined."
# sphinx_gallery_end_ignore

from skore import login

login()

# sphinx_gallery_start_ignore
#
# Delete project before running the example.
from httpx import HTTPStatusError, codes
from skore import Project

try:
    Project.delete(f"{WORKSPACE}/{PROJECT}", mode="hub")
except HTTPStatusError as e:
    if e.response.status_code != codes.NOT_FOUND:
        raise
# sphinx_gallery_end_ignore

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
estimator = tabular_pipeline(LogisticRegression(max_iter=1_000))

# %%
from numpy import logspace
from sklearn.base import clone
from skore import Project, evaluate

project = Project(f"{WORKSPACE}/{PROJECT}", mode="hub")

for regularization in logspace(-3, 3, 5):
    project.put(
        f"lr-regularization-{regularization:.1e}",
        evaluate(
            clone(estimator).set_params(logisticregression__C=regularization),
            X,
            y,
            splitter=0.2,
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
summary.query("log_loss < 0.2")["key"].tolist()

# %%
# Use :meth:`~skore.project._summary.Summary.reports` to load the corresponding
# reports from the project (optionally after filtering the summary).
reports = summary.query("log_loss < 0.2").reports(return_as="comparison")
len(reports.reports_)

# %%
# Since we got a :class:`~skore.ComparisonReport`, we can use the metrics accessor
# to summarize the metrics across the reports.
reports.metrics.summarize().frame()

# %%
reports.metrics.roc().plot(subplot_by=None)

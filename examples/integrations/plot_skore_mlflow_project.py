"""
.. _example_skore_mlflow_project:

==========================================
Store and retrieve Skore reports in MLflow
==========================================

The primilarly goal of `skore` is to create data science artifacts in the form of
structured reports. Those reports can easily be used programmatically via the Python
API. A subsequent aim is to store those reports that you create during your experiment
cycle in a way that it is easy to retrieve them later on.

Skore provides two natives ways to store reports: locally or on Skore Hub. Skore Hub
provides additional interactivity features for you to explore, compare and share
visual insights.

In addition, Skore also provides an MLflow integration to store the content of
reports directly as MLflow artifacts. This example shows how to persist reports in
MLflow using :class:`~skore.Project` in ``mode="mlflow"``: log reports as MLflow runs
and inspect them.

To run this example against your own MLflow tracking server, use:

.. code-block:: bash

    TRACKING_URI=<tracking_uri> PROJECT=<project> python plot_skore_mlflow_project.py

To try it locally, start an MLflow server with ``uvx mlflow server`` and set
``TRACKING_URI=http://127.0.0.1:5000``. For more setup details, see the
`MLflow quickstart <https://mlflow.org/docs/latest/ml/getting-started/running-notebooks/>`_.
"""

# %%
#
# Create a Skore report
# =====================
#
# First, we start by creating a Skore report by evaluating a logistic regression model
# on the iris dataset using some cross-validation.
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skore import evaluate

X, y = load_iris(return_X_y=True, as_frame=True)

estimator = LogisticRegression()
report = evaluate(estimator, X, y, splitter=5)

# %%
#
# Store the Skore reports as MLflow artifacts
# ===========================================
#
# Now, we will store the different items of the Skore report as MLflow artifacts.
# For this matter, you need to create a :class:`~skore.Project` in ``mode="mlflow"``
# and pass the information regarding the MLflow tracking server.

import io

# sphinx_gallery_start_ignore
#
# This part of the code is used for running the example on our continuous integration.
# Configure the context variables and tmp dir:
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from skore import Project

tmp_dir = None

if os.environ.get("SPHINX_BUILD"):
    tmp_dir = TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)
    TRACKING_URI = f"sqlite:///{tmp_path / 'mlflow.db'}"
    PROJECT = "example-skore-mlflow-project"
else:
    assert (TRACKING_URI := os.environ.get("TRACKING_URI")), (
        "`TRACKING_URI` must be defined."
    )
    assert (PROJECT := os.environ.get("PROJECT")), "`PROJECT` must be defined."
# sphinx_gallery_end_ignore

# MLflow/Alembic emits verbose DB initialization logs; silence them so the
# example page focuses on skore usage rather than backend startup details.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    # This creates an MLflow experiment with name `PROJECT`:
    project = Project(
        PROJECT,
        mode="mlflow",
        tracking_uri=TRACKING_URI,
    )

# %%
# Once the project created, the same API used to store a report locally or on Skore Hub
# applies.
project.put("logistic-regression", report)

# sphinx_gallery_start_ignore
if tmp_dir is not None:
    tmp_dir.cleanup()

# sphinx_gallery_end_ignore

# %%
#
# Retrieve the Skore report from MLflow tracking server
# =====================================================
#
# Like for the other modes (local and Skore Hub), you can access what is stored in the
# project via the :meth:`~skore.Project.summarize` method.
import pandas as pd

summary = project.summarize()
pandas_summary = pd.DataFrame(summary).reset_index()
pandas_summary[["id", "key", "report_type", "learner", "ml_task", "dataset"]]

# %%
# Then, you can retrieve a Skore report using the `"id"` column:
(run_id,) = pandas_summary["id"]
loaded_report = project.get(run_id)
loaded_report.metrics.summarize().frame()

# %%
# You can directly use MLflow to access information stored in the MLflow tracking
# server.
import mlflow

mlflow_run = mlflow.get_run(run_id)
mlflow_run.data.metrics

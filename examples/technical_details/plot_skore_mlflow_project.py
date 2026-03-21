"""
.. _example_skore_mlflow_project:

====================
MLflow skore Project
====================

This example shows how to persist reports in MLflow using :class:`~skore.Project`
in ``mode="mlflow"``: log reports as MLflow runs and inspect them. It uses a
:class:`~skore.CrossValidationReport`, but the same approach applies to
:class:`~skore.EstimatorReport`.

To run this example against your own MLflow tracking server, use:

.. code-block:: bash

    TRACKING_URI=<tracking_uri> PROJECT=<project> python plot_skore_mlflow_project.py

To try it locally, start an MLflow server with ``uvx mlflow server`` and set
``TRACKING_URI=http://127.0.0.1:5000``.
"""

# %%
# First, let us build one report to persist:

# %%
from sklearn.datasets import load_iris
from sklearn.ensemble import HistGradientBoostingClassifier
from skore import Project, evaluate

X, y = load_iris(return_X_y=True, as_frame=True)

estimator = HistGradientBoostingClassifier()
report = evaluate(estimator, X, y, splitter=5)

# %%
# Then, we can push the report to the MLflow backend:

import io

# sphinx_gallery_start_ignore
#
# Configure the context variables and tmp dir:
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

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
project.put("hgb-baseline", report)

# sphinx_gallery_start_ignore
if tmp_dir is not None:
    tmp_dir.cleanup()

# sphinx_gallery_end_ignore


# %%
# Note that mlflow warns us about saving models with `pickle`. Future versions of skore
# might rely on `skops <https://skops.readthedocs.io>`__ for model serialization,
# which will make these warnings disappear.

# %%
# Like for other types of projects (local, hub), you can access the summary
# and its DataFrame version:
import pandas as pd

summary = project.summarize()
pandas_summary = pd.DataFrame(summary).reset_index()
pandas_summary[["id", "key", "report_type", "learner", "ml_task", "dataset"]]

# %%
# The "id" column corresponds to the MLflow run ID, so you can access the MLflow
# run this way:
import mlflow

(run_id,) = pandas_summary["id"]

mlflow_run = mlflow.get_run(run_id)
mlflow_run.data.metrics

# %%
# But most importantly, this ID lets you load saved reports:
loaded_report = project.get(run_id)
loaded_report.metrics.summarize().frame()

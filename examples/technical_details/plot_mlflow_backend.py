"""
.. _example_skore_mlflow_project:

==================================
MLflow backend with skore.Project
==================================

This example shows how to persist reports in MLflow using
:class:`~skore.Project` in ``mode="mlflow"``.

This example shows how to use :class:`~skore.Project` in **mlflow** mode: log
reports as MLFlow runs and inspect them.

Examples
--------

To run this example and push in your own MLFlow tracking server, you can run
this example with the following command:

.. code-block:: bash

    TRACKING_URI=<tracking_uri> PROJECT=<project> python plot_skore_hub_project.py

"""
# sphinx_gallery_thumbnail_path = '../../_static/images/screenshot_mlflow_backend.png'

# %%
# Build one report to persist
# ===========================

from sklearn.datasets import load_iris
from sklearn.ensemble import HistGradientBoostingClassifier
from skore import CrossValidationReport, Project

X, y = load_iris(return_X_y=True, as_frame=True)

estimator = HistGradientBoostingClassifier()
report = CrossValidationReport(estimator, X, y)

# %%
# Push the report to the MLflow backend
# =====================================

# sphinx_gallery_start_ignore
#
# Configure the context variables and tmp dir:
import os
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

try:
    # sphinx_gallery_end_ignore

    import io
    from contextlib import redirect_stderr, redirect_stdout

    # MLflow/Alembic emits verbose DB initialization logs; silence them so the
    # example page focuses on skore usage rather than backend startup details.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        project = Project(
            PROJECT,
            mode="mlflow",
            tracking_uri=TRACKING_URI,
        )
        # This creates an MLFlow experiment with name `PROJECT`

    project.put("hgb-baseline", report)

# sphinx_gallery_start_ignore
finally:
    if tmp_dir is not None:
        tmp_dir.cleanup()

# sphinx_gallery_end_ignore


# %%

# You get warnings about serialization, future versions of `skore[mlflow]` might remove
# those warnings by using `skops.io` for models serialization.
#
# .. note::
#    MLflow UI preview for this example:
#
#    .. raw:: html
#
#       <video controls preload="metadata" width="100%" poster="../../_static/images/screenshot_mlflow_backend.png">
#         <source src="../../_static/videos/mlflow_backend_demo.webm" type="video/webm">
#         Your browser does not support the video tag.
#       </video>
#

# %%
# Like for other types of projects (local, hub), you can access the summary
# and it's DataFrame's version:
import pandas as pd

summary = project.summarize()
pandas_summary = pd.DataFrame(summary)
pandas_summary[["key", "report_type", "learner", "ml_task", "dataset"]]

# %%
# The "id" is the MLFlow run ID, you can access the MLFlow run this way if you want:
import mlflow

_, run_id = pandas_summary.index[0]

mlflow_run = mlflow.get_run(run_id)
mlflow_run.data.metrics

# %%
# But most importantly, you can load back a project using this id:
loaded_report = project.get(run_id)
loaded_report.metrics.summarize().frame()

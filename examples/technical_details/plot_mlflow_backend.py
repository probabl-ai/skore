"""
.. _example_skore_mlflow_project:

==================================
MLflow backend with skore.Project
==================================

This example shows how to persist reports in MLflow using
:class:`~skore.Project` in ``mode="mlflow"``.

To keep the example self-contained, we use a temporary SQLite backend store,
push one report, and inspect it through :meth:`~skore.Project.summarize`.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import EstimatorReport, Project, train_test_split
from skrub import tabular_pipeline

# %%
# Build one report to persist
# ===========================
X, y = make_classification(
    n_samples=2_000,
    n_features=20,
    n_informative=8,
    n_redundant=0,
    random_state=42,
)
split_data = train_test_split(
    X=X,
    y=y,
    random_state=42,
    shuffle=False,
    as_dict=True,
)

estimator = tabular_pipeline(LogisticRegression(max_iter=1_000))
report = EstimatorReport(estimator, **split_data)


# %%
# Push the report to the MLflow backend
# =====================================
#
# We use a temporary SQLite backend store so the example is fully self-contained.
# The same code also works with a running MLflow server URI
# (e.g. ``tracking_uri="http://127.0.0.1:5000"``).
with TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    backend_store_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    # MLflow/Alembic emits verbose DB initialization logs; silence them so the
    # example page focuses on skore usage rather than backend startup details.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        project = Project(
            "technical-details-mlflow",
            mode="mlflow",
            tracking_uri=backend_store_uri,
        )
    project.put("logreg-baseline", report)
    summary = project.summarize()
    summary

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

from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from skore import EstimatorReport, Project, train_test_split
from skrub import tabular_pipeline

# %%
# Build one report to persist
# ===========================
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
split_data = train_test_split(
    X=X,
    y=y,
    random_state=42,
    shuffle=False,
    as_dict=True,
)

estimator = tabular_pipeline(DecisionTreeRegressor(max_depth=5))
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
    project.put("linreg-baseline", report)
    summary = project.summarize()

# %%
# To see the normal DataFrame table instead of the widget (e.g. in scripts or
# when you prefer the table), wrap the summary in :class:`pandas.DataFrame`:
import pandas as pd

pandas_summary = pd.DataFrame(summary)
pandas_summary

import os
import shutil
from pathlib import Path

from pytest import fixture

# TODO: remove once I finished iterating with codex.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from skore import EstimatorReport


@fixture(scope="session", autouse=True)
def cleanup_mlruns():
    yield
    shutil.rmtree(Path.cwd() / "mlruns", ignore_errors=True)


@fixture(scope="module")
def regression() -> EstimatorReport:
    X, y = make_regression(random_state=42, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return EstimatorReport(
        Ridge(random_state=42),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

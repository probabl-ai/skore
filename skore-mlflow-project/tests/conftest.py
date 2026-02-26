import shutil
from pathlib import Path

from pytest import fixture
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from skore import CrossValidationReport, EstimatorReport


@fixture(scope="session", autouse=True)
def cleanup_mlruns():
    yield
    shutil.rmtree(Path.cwd() / "mlruns", ignore_errors=True)


@fixture(scope="module")
def regression() -> EstimatorReport:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
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


@fixture(scope="module")
def regression_cv() -> CrossValidationReport:
    X, y = load_iris(return_X_y=True)
    return CrossValidationReport(
        DecisionTreeClassifier(max_depth=3, random_state=42), X=X, y=y, splitter=2
    )

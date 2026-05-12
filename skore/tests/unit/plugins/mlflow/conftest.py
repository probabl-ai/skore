import mlflow
import pandas as pd
from pytest import fixture
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_linnerud,
    load_wine,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skore import CrossValidationReport, EstimatorReport


@fixture(autouse=True)
def isolated_mlflow_tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    try:
        yield tracking_uri
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(previous_tracking_uri)


@fixture(scope="module")
def reg_report() -> EstimatorReport:
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
def clf_report() -> EstimatorReport:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        DecisionTreeClassifier(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture(scope="module")
def mclf_report() -> EstimatorReport:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        DecisionTreeClassifier(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture(scope="module")
def mreg_report() -> EstimatorReport:
    X, y = load_linnerud(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        DecisionTreeRegressor(random_state=0, max_depth=5),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture(scope="module")
def cv_clf_report() -> CrossValidationReport:
    """binary-classification"""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return CrossValidationReport(
        DecisionTreeClassifier(random_state=0), X, y, splitter=2
    )


@fixture(scope="module")
def cv_mclf_report() -> CrossValidationReport:
    """multiclass-classification"""
    X, y = load_wine(return_X_y=True)
    return CrossValidationReport(
        DecisionTreeClassifier(max_depth=3, random_state=42), X, y, splitter=2
    )


@fixture(scope="module")
def cv_reg_report() -> CrossValidationReport:
    """regression"""
    X, y = load_diabetes(return_X_y=True)
    return CrossValidationReport(
        DecisionTreeRegressor(random_state=0, max_depth=5), X, y, splitter=2
    )


@fixture(scope="module")
def cv_mreg_report() -> CrossValidationReport:
    """multiouput-regression"""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y: pd.Series
    multi_target_y = pd.concat(
        [y, y + y.sample(len(y))],
        axis=1,
    )
    return CrossValidationReport(
        DecisionTreeRegressor(random_state=0, max_depth=5),
        X,
        multi_target_y,
        splitter=2,
    )

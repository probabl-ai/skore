import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_linnerud,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skore import CrossValidationReport, EstimatorReport

from skore_mlflow_project.reports import (
    Artifact,
    Metric,
    Model,
    Params,
    Tag,
    iter_cv,
    iter_cv_metrics,
    iter_estimator,
    iter_estimator_metrics,
)

REPORT_FIXTURES = ["clf_report", "mclf_report", "reg_report", "mreg_report"]
CV_REPORT_FIXTURES = [
    "cv_clf_report",
    "cv_mclf_report",
    "cv_reg_report",
    "cv_mreg_report",
]


@pytest.fixture(scope="module")
def clf_report() -> EstimatorReport:
    """binary-classification"""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        DecisionTreeClassifier(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def mclf_report() -> EstimatorReport:
    """multiclass-classification"""
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        DecisionTreeClassifier(random_state=0),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def reg_report() -> EstimatorReport:
    """regression"""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        DecisionTreeRegressor(random_state=0, max_depth=5),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def mreg_report() -> EstimatorReport:
    """multiouput-regression"""
    X, y = load_linnerud(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return EstimatorReport(
        DecisionTreeRegressor(random_state=0, max_depth=5),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def cv_clf_report() -> CrossValidationReport:
    """binary-classification"""
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return CrossValidationReport(
        DecisionTreeClassifier(random_state=0), X, y, splitter=2
    )


@pytest.fixture(scope="module")
def cv_mclf_report() -> CrossValidationReport:
    """multiclass-classification"""
    X, y = load_wine(return_X_y=True)
    return CrossValidationReport(
        DecisionTreeClassifier(random_state=0), X, y, splitter=2
    )


@pytest.fixture(scope="module")
def cv_reg_report() -> CrossValidationReport:
    """regression"""
    X, y = load_diabetes(return_X_y=True)
    return CrossValidationReport(
        DecisionTreeRegressor(random_state=0, max_depth=5), X, y, splitter=2
    )


@pytest.fixture(scope="module")
def cv_mreg_report() -> CrossValidationReport:
    """multiouput-regression"""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    multi_target_y = pd.concat(
        [y, y + np.random.default_rng(0).normal(0, 20, len(y))],
        axis=1,
    )
    return CrossValidationReport(
        DecisionTreeRegressor(random_state=0, max_depth=5),
        X,
        multi_target_y,
        splitter=2,
    )


@pytest.fixture
def report(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("report", REPORT_FIXTURES, indirect=True)
def test_iter_estimator_metrics_smoke(report):
    assert all(
        isinstance(obj, Artifact | Metric) for obj in iter_estimator_metrics(report)
    )


@pytest.mark.parametrize("report", CV_REPORT_FIXTURES, indirect=True)
def test_iter_cv_metrics_smoke(report):
    assert all(isinstance(obj, Artifact | Metric) for obj in iter_cv_metrics(report))


@pytest.mark.parametrize("report", REPORT_FIXTURES, indirect=True)
def test_iter_estimator_smoke(report):
    assert all(
        isinstance(obj, Artifact | Metric | Params | Tag | Model)
        for obj in iter_estimator(report)
    )


@pytest.mark.parametrize("report", CV_REPORT_FIXTURES, indirect=True)
def test_iter_cv_smoke(report):
    assert all(
        isinstance(obj, Artifact | Metric | Params | Tag | Model | tuple)
        for obj in iter_cv(report)
    )

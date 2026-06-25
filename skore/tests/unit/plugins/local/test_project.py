from pathlib import Path

import pandas as pd
import pytest
from pytest import fixture
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from skore import CrossValidationReport, EstimatorReport
from skore._plugins.local import Project


@fixture(scope="module")
def regression() -> EstimatorReport:
    X, y = make_regression(random_state=42)
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
def regression_dummy(regression) -> EstimatorReport:
    return EstimatorReport(
        DummyRegressor(),
        X_train=regression.X_train,
        y_train=regression.y_train,
        X_test=regression.X_test,
        y_test=regression.y_test,
    )


@fixture(scope="module")
def cv_regression() -> CrossValidationReport:
    X, y = make_regression(random_state=42)

    return CrossValidationReport(Ridge(random_state=42), X, y)


@fixture(scope="module")
def binary_classification() -> EstimatorReport:
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return EstimatorReport(
        RandomForestClassifier(random_state=42),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@fixture(scope="module")
def cv_binary_classification() -> CrossValidationReport:
    X, y = make_classification(random_state=42, n_samples=10)

    return CrossValidationReport(
        RandomForestClassifier(random_state=42), X, y, splitter=2
    )


def test_init_delete(tmp_path):
    project = Project(name="regression", workspace=tmp_path)
    Project(name="regression_1", workspace=tmp_path)
    assert (tmp_path / "projects" / "regression").exists()
    project.delete(name="regression", workspace=tmp_path)
    assert not (tmp_path / "projects" / "regression").exists()
    assert (tmp_path / "projects" / "regression_1").exists()


def test_put_get_summarize(tmp_path, regression, regression_dummy, cv_regression):
    project = Project(name="regression", workspace=tmp_path)
    project.put("ridge", regression)
    project.put("dummy", regression_dummy)
    project.put("cv", cv_regression)
    fetched_regression = project.get(str(regression.id))
    fetched_dummy = project.get(regression_dummy.id)
    fetched_cv = project.get(cv_regression.id)
    assert ("test", "predict", None) in fetched_regression._cache
    assert (
        "test",
        "r2",
        ("mapping", (("multioutput", "raw_values"),)),
    ) in fetched_regression._cache
    assert fetched_regression.metrics.get("r2") == regression.metrics.get("r2")
    assert (fetched_regression.X_train == regression.X_train).all()
    assert fetched_dummy.metrics.get("r2") == regression_dummy.metrics.get("r2")
    pd.testing.assert_frame_equal(
        fetched_cv.metrics.summarize().frame(),
        cv_regression.metrics.summarize().frame(),
    )
    summary = project.summarize()
    assert len(summary) == 3
    assert Path(next(iter(summary))["local_path"]).is_relative_to(tmp_path)


def test_permutation_importances(tmp_path, regression_dummy):
    project = Project(name="regression", workspace=tmp_path)
    importances = regression_dummy.inspection.permutation_importance().frame()
    project.put("regression", regression_dummy)
    fetched = project.get(regression_dummy.id)
    assert any(k[:2] == ("test", "permutation_importance") for k in fetched._cache)
    pd.testing.assert_frame_equal(
        fetched.inspection.permutation_importance().frame(), importances
    )


def test_init_with_envar(monkeypatch, tmp_path):
    monkeypatch.setenv("SKORE_WORKSPACE", str(tmp_path))
    project = Project("<project>")
    assert project.name == "_project_"
    assert project.path == tmp_path / "projects" / "_project_"


@pytest.mark.parametrize("type", [str, Path])
def test_init_with_workspace(tmp_path, type):
    project = Project("<project>", workspace=type(tmp_path))
    assert project.path == tmp_path / "projects" / "_project_"


def test_find_workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    env = tmp_path / "env"
    local = tmp_path / "repo"
    pwd = local / "a" / "b"
    for d in (home, env, pwd):
        d.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.chdir(pwd)
    assert Project("regression").path == home / "skore_data" / "projects" / "regression"
    Project("abc", workspace=local / "skore_data")
    assert (
        Project("regression").path == local / "skore_data" / "projects" / "regression"
    )
    monkeypatch.setenv("SKORE_WORKSPACE", str(env))
    assert Project("regression").path == env / "projects" / "regression"


def test_get_missing(tmp_path):
    p = Project("regression", workspace=tmp_path)
    with pytest.raises(KeyError, match="17"):
        p.get(17)

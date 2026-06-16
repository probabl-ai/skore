from pathlib import Path

import joblib
import pytest
from pandas import DataFrame
from pytest import fixture, raises
from sklearn.datasets import make_classification, make_regression
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


@fixture(autouse=True)
def monkeypatch_datetime(monkeypatch, Datetime):
    pass


@fixture(autouse=True)
def monkeypatch_metrics(monkeypatch, Datetime):
    monkeypatch.setattr(
        "skore.EstimatorReport.metrics.rmse",
        lambda _, data_source: float(hash(f"<rmse_{data_source}>")),
    )
    monkeypatch.setattr(
        "skore.EstimatorReport.metrics.timings",
        lambda _: {
            "fit_time": float(hash("<fit_time>")),
            "predict_time_train": float(hash("<predict_time_train>")),
            "predict_time_test": float(hash("<predict_time_test>")),
        },
    )
    monkeypatch.setattr(
        "skore.CrossValidationReport.metrics.rmse",
        lambda _, data_source, aggregate: DataFrame.from_dict(
            {
                ("Ridge", "mean"): {"RMSE": float(hash(f"<rmse_mean_{data_source}>"))},
                ("Ridge", "std"): {"RMSE": float(hash(f"<rmse_std_{data_source}>"))},
            }
        ),
    )
    monkeypatch.setattr(
        "skore.CrossValidationReport.metrics.timings",
        lambda _, aggregate: DataFrame.from_dict(
            {
                "mean": {
                    "Fit time (s)": float(hash("<fit_time_mean>")),
                    "Predict time train (s)": float(hash("<predict_time_mean_train>")),
                    "Predict time test (s)": float(hash("<predict_time_mean_test>")),
                },
                "std": {
                    "Fit time (s)": float(hash("<fit_time_std>")),
                    "Predict time train (s)": float(hash("<predict_time_std_train>")),
                    "Predict time test (s)": float(hash("<predict_time_std_test>")),
                },
            }
        ),
    )


class TestProject:
    def test_init(self, tmp_path):
        project = Project("<project>")
        assert project.name == "<project>"

    def test_put_estimator_report_reuses_artifact_id(self, tmp_path, regression):
        project = Project("<project>", workspace=tmp_path)

        project.put("<key-1>", regression)
        regression.cache_predictions()
        project.put("<key-2>", regression)

        assert len(project.summarize()) == 2

        # Make sure the pickle is not broken:
        report = project.get(str(regression.id))
        report.cache_predictions()

    def test_put_cross_validation_report_reuses_artifact_id(
        self, tmp_path, cv_regression
    ):
        project = Project("<project>", workspace=tmp_path)

        project.put("<key-1>", cv_regression)
        cv_regression.cache_predictions()
        project.put("<key-2>", cv_regression)

        assert len(project.summarize()) == 2

        # Make sure the pickle is not broken:
        report = project.get(str(cv_regression.id))
        report.cache_predictions()

    def test_init_with_envar(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SKORE_WORKSPACE", str(tmp_path))
        project = Project("<project>")

        assert project.name == "<project>"

    @pytest.mark.parametrize("type", [str, Path])
    def test_init_with_workspace(self, tmp_path, type):
        project = Project("<project>", workspace=type(tmp_path))

        assert project.name == "<project>"

    def test_put_exception(self, tmp_path, regression):
        import re

        project = Project("<project>", workspace=tmp_path)

        with raises(TypeError, match="Key must be a string"):
            project.put(None, "<value>")

        with raises(TypeError, match="Report must be a `skore.EstimatorReport` or"):
            project.put("<key>", "<value>")

        Project.delete("<project>", workspace=tmp_path)

        with raises(
            RuntimeError,
            match=re.escape(
                f"Skore could not proceed because "
                f"Project(mode='local', name='<project>', workspace='{tmp_path}') "
                f"does not exist anymore."
            ),
        ):
            project.put("<key>", regression)

    def test_put_estimator_report(self, tmp_path, nowstr, regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", regression)

    def test_put_cross_validation_report(self, tmp_path, nowstr, cv_regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", cv_regression)

        project.put("<key>", cv_regression)

    def test_get(self, tmp_path, regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", regression)
        project.put("<key>", regression)

        report = project.get(str(regression.id))

        assert isinstance(report, EstimatorReport)
        assert report.estimator_name_ == regression.estimator_name_
        assert report._ml_task == regression._ml_task

    def test_get_exception(self, tmp_path, regression):
        import re

        project = Project("<project>", workspace=tmp_path)
        Project.delete("<project>", workspace=tmp_path)

        with raises(
            RuntimeError,
            match=re.escape(
                f"Skore could not proceed because "
                f"Project(mode='local', name='<project>', workspace='{tmp_path}') "
                f"does not exist anymore."
            ),
        ):
            project.get(None)

    def test_summarize(self, tmp_path, Datetime, regression, cv_regression):
        project = Project("<project>", workspace=tmp_path)

        project.put("<key1>", regression)
        project.put("<key1>", regression)
        project.put("<key2>", cv_regression)

        assert project.summarize() == [
            {
                "id": str(regression.id),
                "key": "<key1>",
                "date": Datetime.nows_isoformat[0],
                "learner": "Ridge",
                "ml_task": "regression",
                "report_type": "estimator",
                "dataset": joblib.hash(regression.y_test),
                "rmse": float(hash("<rmse_test>")),
                "log_loss": None,
                "roc_auc": None,
                "fit_time": float(hash("<fit_time>")),
                "predict_time": float(hash("<predict_time_test>")),
                "rmse_mean": None,
                "log_loss_mean": None,
                "roc_auc_mean": None,
                "fit_time_mean": None,
                "predict_time_mean": None,
                "rmse_std": None,
                "log_loss_std": None,
                "roc_auc_std": None,
                "fit_time_std": None,
                "predict_time_std": None,
            },
            {
                "id": str(regression.id),
                "key": "<key1>",
                "date": Datetime.nows_isoformat[1],
                "learner": "Ridge",
                "ml_task": "regression",
                "report_type": "estimator",
                "dataset": joblib.hash(regression.y_test),
                "rmse": float(hash("<rmse_test>")),
                "log_loss": None,
                "roc_auc": None,
                "fit_time": float(hash("<fit_time>")),
                "predict_time": float(hash("<predict_time_test>")),
                "rmse_mean": None,
                "log_loss_mean": None,
                "roc_auc_mean": None,
                "fit_time_mean": None,
                "predict_time_mean": None,
                "rmse_std": None,
                "log_loss_std": None,
                "roc_auc_std": None,
                "fit_time_std": None,
                "predict_time_std": None,
            },
            {
                "id": str(cv_regression.id),
                "key": "<key2>",
                "date": Datetime.nows_isoformat[2],
                "learner": "Ridge",
                "ml_task": "regression",
                "report_type": "cross-validation",
                "dataset": joblib.hash(cv_regression.y),
                "rmse": None,
                "log_loss": None,
                "roc_auc": None,
                "fit_time": None,
                "predict_time": None,
                "rmse_mean": float(hash("<rmse_mean_test>")),
                "log_loss_mean": None,
                "roc_auc_mean": None,
                "fit_time_mean": float(hash("<fit_time_mean>")),
                "predict_time_mean": float(hash("<predict_time_mean_test>")),
                "rmse_std": float(hash("<rmse_std_test>")),
                "log_loss_std": None,
                "roc_auc_std": None,
                "fit_time_std": float(hash("<fit_time_std>")),
                "predict_time_std": float(hash("<predict_time_std_test>")),
            },
        ]

    def test_summarize_exception(self, tmp_path, regression):
        import re

        project = Project("<project>", workspace=tmp_path)
        Project.delete("<project>", workspace=tmp_path)

        with raises(
            RuntimeError,
            match=re.escape(
                f"Skore could not proceed because "
                f"Project(mode='local', name='<project>', workspace='{tmp_path}') "
                f"does not exist anymore."
            ),
        ):
            project.summarize()

    def test_delete(self, tmp_path, binary_classification, regression):
        project1 = Project("<project1>", workspace=tmp_path)
        project1.put("<project1-key1>", binary_classification)
        project1.put("<project1-key2>", regression)

        project2 = Project("<project2>", workspace=tmp_path)
        project2.put("<project2-key1>", binary_classification)

        Project.delete("<project1>", workspace=tmp_path)

    def test_delete_exception(self, tmp_path):
        import re

        with raises(
            LookupError,
            match=re.escape(
                f"Project(mode='local', name='<project>', workspace='{tmp_path}') "
                f"does not exist."
            ),
        ):
            Project.delete("<project>", workspace=tmp_path)

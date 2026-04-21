import joblib
import numpy as np
import pytest
from pandas import DataFrame
from pytest import fixture, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from skore import CrossValidationReport, EstimatorReport, evaluate

from skore_local_project import Project
from skore_local_project.storage import DiskCacheStorage


class CountingRidge(Ridge):
    predict_calls = 0

    @classmethod
    def reset_predict_calls(cls) -> None:
        cls.predict_calls = 0

    def predict(self, X):
        type(self).predict_calls += 1
        return super().predict(X)


@fixture
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


@fixture
def cv_regression() -> CrossValidationReport:
    X, y = make_regression(random_state=42)

    return CrossValidationReport(Ridge(random_state=42), X, y)


@fixture
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


@fixture
def cv_binary_classification() -> CrossValidationReport:
    X, y = make_classification(random_state=42, n_samples=10)

    return CrossValidationReport(
        RandomForestClassifier(random_state=42), X, y, splitter=2
    )


@fixture(autouse=True)
def monkeypatch_datetime(monkeypatch, Datetime):
    monkeypatch.setattr("skore_local_project.metadata.datetime", Datetime)


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
                ("Ridge", "mean"): {
                    "RMSE": float(hash(f"<rmse_{aggregate}_{data_source}>"))
                }
            }
        ),
    )
    monkeypatch.setattr(
        "skore.CrossValidationReport.metrics.timings",
        lambda _, aggregate: DataFrame.from_dict(
            {
                "mean": {
                    "Fit time (s)": float(hash(f"<fit_time_{aggregate}>")),
                    "Predict time train (s)": float(
                        hash(f"<predict_time_{aggregate}_train>")
                    ),
                    "Predict time test (s)": float(
                        hash(f"<predict_time_{aggregate}_test>")
                    ),
                }
            }
        ),
    )


class TestProject:
    def test_init(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "skore_local_project.project.platformdirs.user_data_dir",
            lambda: str(tmp_path),
        )

        project = Project("<project>")

        assert project.workspace == (tmp_path / "skore")
        assert project.name == "<project>"
        assert isinstance(project._Project__metadata_storage, DiskCacheStorage)
        assert isinstance(project._Project__artifacts_storage, DiskCacheStorage)

    def test_put_estimator_reports_deduplicate_data(self, tmp_path, regression):
        project = Project("<project>", workspace=tmp_path)
        other_regression = EstimatorReport(
            Ridge(alpha=2.0, random_state=42),
            X_train=regression.X_train,
            y_train=regression.y_train,
            X_test=regression.X_test,
            y_test=regression.y_test,
        )

        project.put("<key-1>", regression)
        project.put("<key-2>", other_regression)

        # Ensure only one data artifact was persisted:
        n_data = sum(
            key.startswith("data_") for key in project._Project__artifacts_storage
        )
        assert n_data == 1
        # but two different reports:
        summaries = project.summarize()
        assert len(summaries) == 2
        artifact_ids = {metadata["id"] for metadata in summaries}
        assert len(artifact_ids) == 2

        # Make sure the pickle is not broken:
        reports = [project.get(metadata["id"]) for metadata in summaries]
        assert all(isinstance(report, EstimatorReport) for report in reports)
        assert [report.estimator_.alpha for report in reports] == [1.0, 2.0]

    def test_put_cross_validation_reports_deduplicate_data(
        self, tmp_path, cv_regression
    ):
        project = Project("<project>", workspace=tmp_path)
        other_cv_regression = CrossValidationReport(
            Ridge(alpha=2.0, random_state=42),
            cv_regression.X,
            cv_regression.y,
        )

        project.put("<key-1>", cv_regression)
        project.put("<key-2>", other_cv_regression)

        # Ensure only one data artifact was persisted:
        n_data = sum(
            key.startswith("data_") for key in project._Project__artifacts_storage
        )
        assert n_data == 1
        # but two different reports:
        summaries = project.summarize()
        assert len(summaries) == 2
        artifact_ids = {metadata["id"] for metadata in summaries}
        assert len(artifact_ids) == 2

        # Make sure the pickle is not broken:
        reports = [project.get(metadata["id"]) for metadata in summaries]
        assert all(isinstance(report, CrossValidationReport) for report in reports)
        assert [report.estimator_.alpha for report in reports] == [1.0, 2.0]

    def test_init_with_envar(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SKORE_WORKSPACE", str(tmp_path))
        project = Project("<project>")

        assert project.workspace == tmp_path
        assert project.name == "<project>"
        assert isinstance(project._Project__metadata_storage, DiskCacheStorage)
        assert isinstance(project._Project__artifacts_storage, DiskCacheStorage)

    def test_init_with_workspace(self, tmp_path):
        project = Project("<project>", workspace=tmp_path)

        assert project.workspace == tmp_path
        assert project.name == "<project>"
        assert isinstance(project._Project__metadata_storage, DiskCacheStorage)
        assert isinstance(project._Project__artifacts_storage, DiskCacheStorage)

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

        # Ensure artifacts are persisted
        n_artifacts = len(project._Project__artifacts_storage)
        assert n_artifacts >= 1

        # Ensure metadata are persisted:
        assert len(project.summarize()) == 1
        (metadata,) = project.summarize()
        assert metadata["key"] == "<key>"
        assert metadata["date"] == nowstr
        assert metadata["learner"] == "Ridge"
        assert metadata["dataset"] == joblib.hash(regression.y_test)
        assert metadata["ml_task"] == "regression"
        assert metadata["report_type"] == "estimator"
        assert metadata["rmse"] == float(hash("<rmse_test>"))
        assert metadata["log_loss"] is None
        assert metadata["roc_auc"] is None
        assert metadata["fit_time"] == float(hash("<fit_time>"))
        assert metadata["predict_time"] == float(hash("<predict_time_test>"))

        # Ensure put twice (even with the same key)
        project.put("<key>", regression)
        assert len(project.summarize()) == 2
        # Ensure artifacts are reused
        assert n_artifacts == len(project._Project__artifacts_storage)

        # Put the same report with a different key:
        project.put("<key-2>", regression)
        assert len(project.summarize()) == 3
        *_, metadata = project.summarize()
        assert metadata["key"] == "<key-2>"
        # Ensure artifacts are reused
        assert n_artifacts == len(project._Project__artifacts_storage)

        # Do some calculation (mutates the report and hence the state):
        regression.cache_predictions()
        project.put("<key-3>", regression)
        # Ensure some new artifacts were created, but some were reused
        new_n_artifacts = len(project._Project__artifacts_storage)
        assert n_artifacts < new_n_artifacts < 2 * new_n_artifacts

        # Ensure nothing is returned for other projects
        project = Project("<other-project>", workspace=tmp_path)
        assert len(project.summarize()) == 0

    def test_put_cross_validation_report(self, tmp_path, nowstr, cv_regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", cv_regression)

        # Ensure artifacts are persisted
        n_artifacts = len(project._Project__artifacts_storage)
        assert n_artifacts >= 1

        # Ensure metadata are persisted:
        assert len(project.summarize()) == 1
        (metadata,) = project.summarize()
        assert metadata["key"] == "<key>"
        assert metadata["date"] == nowstr
        assert metadata["learner"] == "Ridge"
        assert metadata["dataset"] == joblib.hash(cv_regression.y)
        assert metadata["ml_task"] == "regression"
        assert metadata["report_type"] == "cross-validation"
        assert metadata["rmse_mean"] == float(hash("<rmse_mean_test>"))
        assert metadata["log_loss_mean"] is None
        assert metadata["roc_auc_mean"] is None
        assert metadata["fit_time_mean"] == float(hash("<fit_time_mean>"))
        assert metadata["predict_time_mean"] == float(hash("<predict_time_mean_test>"))

        # Ensure put twice (even with the same key)
        project.put("<key>", cv_regression)
        assert len(project.summarize()) == 2
        # A second put materializes the stable estimator_reports state
        assert n_artifacts == len(project._Project__artifacts_storage)

        # Put the same report with a different key:
        project.put("<key-2>", cv_regression)
        assert len(project.summarize()) == 3
        *_, metadata = project.summarize()
        assert metadata["key"] == "<key-2>"
        # Ensure artifacts are reused
        assert n_artifacts == len(project._Project__artifacts_storage)

        # Do some calculation (mutates the report and hence the state):
        cv_regression.cache_predictions()
        project.put("<key-3>", cv_regression)
        # Ensure some new artifacts were created, but some were reused
        new_n_artifacts = len(project._Project__artifacts_storage)
        assert n_artifacts < new_n_artifacts < n_artifacts * 2

        # Ensure nothing is returned for other projects
        project = Project("<other-project>", workspace=tmp_path)
        assert len(project.summarize()) == 0

    @pytest.mark.parametrize("splitter", [0.2, 5])
    def test_get_restores_cached_predictions(self, tmp_path, splitter):
        X, y = make_regression(random_state=42)
        report0 = evaluate(CountingRidge(), X, y, splitter=splitter)
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", report0)

        # Ensure the persisted report can be restored:
        artifact_id = project.summarize()[-1]["id"]
        report1 = project.get(artifact_id)
        assert isinstance(report1, report0.__class__)
        assert report1.estimator_name_ == report0.estimator_name_
        assert report1._ml_task == report0._ml_task

        # The first restored report has no cached train predictions yet:
        CountingRidge.reset_predict_calls()
        report1.get_predictions(data_source="train")
        assert CountingRidge.predict_calls >= 1

        # Persist the report again after computing train predictions:
        project.put("<key>", report1)
        artifact_id = project.summarize()[-1]["id"]
        report2 = project.get(artifact_id)

        # Ensure cached train predictions are restored:
        CountingRidge.reset_predict_calls()
        report2.get_predictions(data_source="train")
        assert CountingRidge.predict_calls == 0

        # Ensure the restored predictions match the original report:
        predictions2 = report2.get_predictions(data_source="train")
        predictions0 = report0.get_predictions(data_source="train")
        if isinstance(predictions2, list):
            for y_pred2, y_pred0 in zip(predictions2, predictions0, strict=True):
                np.testing.assert_array_equal(y_pred2, y_pred0)
        else:
            np.testing.assert_array_equal(predictions2, predictions0)

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

    def test_summarize_exception(self, tmp_path):
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

    def test_delete(self, tmp_path, binary_classification, regression, cv_regression):
        project1 = Project("<project1>", workspace=tmp_path)
        project1.put("<key>", regression)
        n_artifacts = len(DiskCacheStorage(tmp_path / "artifacts"))
        assert n_artifacts >= 1

        project2 = Project("<project2>", workspace=tmp_path)
        project2.put("<key>", binary_classification)
        project2.put("<key>", cv_regression)
        regression.cache_predictions()
        project2.put("<key>", regression)

        assert len(DiskCacheStorage(tmp_path / "metadata")) == 4
        assert len(DiskCacheStorage(tmp_path / "artifacts")) > n_artifacts

        Project.delete("<project2>", workspace=tmp_path)

        project1 = Project("<project1>", workspace=tmp_path)
        assert len(project1.summarize()) == 1
        assert len(DiskCacheStorage(tmp_path / "metadata")) == 1
        assert len(DiskCacheStorage(tmp_path / "artifacts")) == n_artifacts

        artifact_id = project1.summarize()[-1]["id"]
        report = project1.get(artifact_id)
        assert isinstance(report, EstimatorReport)

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

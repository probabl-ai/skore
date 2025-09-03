from io import BytesIO
from types import SimpleNamespace

import joblib
from pandas import DataFrame
from pytest import fixture, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from skore import CrossValidationReport, EstimatorReport
from skore_local_project import Project
from skore_local_project.storage import DiskCacheStorage


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
            "skore_local_project.project.platformdirs.user_cache_dir",
            lambda: str(tmp_path),
        )

        project = Project("<project>")

        assert project.workspace == (tmp_path / "skore")
        assert project.name == "<project>"
        assert project.run_id
        assert isinstance(project._Project__metadata_storage, DiskCacheStorage)
        assert isinstance(project._Project__artifacts_storage, DiskCacheStorage)

    def test_pickle_estimator_report(self, regression):
        # Pickle the report once, without any value in the cache
        assert not regression._cache
        pickle_1 = Project.pickle(regression)
        assert not regression._cache

        # Pickle the same report, but with values in the cache
        regression.cache_predictions()

        assert regression._cache
        pickle_2 = Project.pickle(regression)
        assert regression._cache

        # Make sure that the two pickles on the report are not affected by the cache
        assert pickle_1 == pickle_2

    def test_pickle_cross_validation_report(self, cv_regression):
        reports = [cv_regression] + cv_regression.estimator_reports_

        # Pickle the report once, without any value in the cache
        assert not any(report._cache for report in reports)
        pickle_1 = Project.pickle(cv_regression)
        assert not any(report._cache for report in reports)

        # Pickle the same report, but with values in the cache
        cv_regression.cache_predictions()

        assert any(report._cache for report in reports)
        pickle_2 = Project.pickle(cv_regression)
        assert any(report._cache for report in reports)

        # Make sure that the two pickles on the report are not affected by the cache
        assert pickle_1 == pickle_2

    def test_init_with_envar(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SKORE_WORKSPACE", str(tmp_path))
        project = Project("<project>")

        assert project.workspace == tmp_path
        assert project.name == "<project>"
        assert project.run_id
        assert isinstance(project._Project__metadata_storage, DiskCacheStorage)
        assert isinstance(project._Project__artifacts_storage, DiskCacheStorage)

    def test_init_with_workspace(self, tmp_path):
        project = Project("<project>", workspace=tmp_path)

        assert project.workspace == tmp_path
        assert project.name == "<project>"
        assert project.run_id
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
                f"Bad condition: {repr(project)} does not exist anymore, "
                f"it had to be removed.",
            ),
        ):
            project.put("<key>", regression)

    def test_put_estimator_report(self, tmp_path, nowstr, regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", regression)

        # Ensure artifacts are persisted
        assert len(project._Project__artifacts_storage) == 1

        with BytesIO(next(project._Project__artifacts_storage.values())) as stream:
            artifact = joblib.load(stream)

        assert isinstance(artifact, EstimatorReport)
        assert artifact.estimator_name_ == "Ridge"
        assert artifact.ml_task == "regression"

        # Ensure metadata are persisted
        assert len(project._Project__metadata_storage) == 1
        assert list(project._Project__metadata_storage.values()) == [
            {
                "project_name": "<project>",
                "run_id": project.run_id,
                "key": "<key>",
                "artifact_id": next(project._Project__artifacts_storage.keys()),
                "date": nowstr,
                "learner": "Ridge",
                "dataset": joblib.hash(regression.y_test),
                "ml_task": "regression",
                "report_type": "estimator",
                "rmse": float(hash("<rmse_test>")),
                "log_loss": None,
                "roc_auc": None,
                "fit_time": float(hash("<fit_time>")),
                "predict_time": float(hash("<predict_time_test>")),
            }
        ]

        # Ensure put twice
        project.put("<key>", regression)

        assert len(project._Project__artifacts_storage) == 1
        assert len(project._Project__metadata_storage) == 2

    def test_put_cross_validation_report(self, tmp_path, nowstr, cv_regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", cv_regression)

        # Ensure artifacts are persisted
        assert len(project._Project__artifacts_storage) == 1

        with BytesIO(next(project._Project__artifacts_storage.values())) as stream:
            artifact = joblib.load(stream)

        assert isinstance(artifact, CrossValidationReport)
        assert artifact.estimator_name_ == cv_regression.estimator_name_
        assert artifact.ml_task == "regression"

        # Ensure metadata are persisted
        assert len(project._Project__metadata_storage) == 1
        assert list(project._Project__metadata_storage.values()) == [
            {
                "project_name": "<project>",
                "run_id": project.run_id,
                "key": "<key>",
                "artifact_id": next(project._Project__artifacts_storage.keys()),
                "date": nowstr,
                "learner": "Ridge",
                "dataset": joblib.hash(cv_regression.y),
                "ml_task": "regression",
                "report_type": "cross-validation",
                "rmse_mean": float(hash("<rmse_mean_test>")),
                "log_loss_mean": None,
                "roc_auc_mean": None,
                "fit_time_mean": float(hash("<fit_time_mean>")),
                "predict_time_mean": float(hash("<predict_time_mean_test>")),
            }
        ]

        # Ensure put twice
        project.put("<key>", cv_regression)

        assert len(project._Project__artifacts_storage) == 1
        assert len(project._Project__metadata_storage) == 2

    def test_reports(self, tmp_path):
        project = Project("<project>", workspace=tmp_path)

        assert isinstance(project.reports, SimpleNamespace)
        assert hasattr(project.reports, "get")
        assert hasattr(project.reports, "metadata")

    def test_reports_exception(self, tmp_path):
        import re

        project = Project("<project>", workspace=tmp_path)
        Project.delete("<project>", workspace=tmp_path)

        with raises(
            RuntimeError,
            match=re.escape(
                f"Bad condition: {repr(project)} does not exist anymore, "
                f"it had to be removed.",
            ),
        ):
            project.reports  # noqa: B018

    def test_reports_get(self, tmp_path, regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", regression)
        project.put("<key>", regression)

        report = project.reports.get(next(project._Project__artifacts_storage.keys()))

        assert len(project._Project__artifacts_storage) == 1
        assert len(project._Project__metadata_storage) == 2
        assert isinstance(report, EstimatorReport)
        assert report.estimator_name_ == regression.estimator_name_
        assert report._ml_task == regression._ml_task

    def test_reports_metadata(self, tmp_path, nowstr, regression):
        project = Project("<project>", workspace=tmp_path)

        project.put("<key>", regression)
        project.put("<key>", regression)

        assert len(project._Project__artifacts_storage) == 1
        assert len(project._Project__metadata_storage) == 2
        assert project.reports.metadata() == [
            {
                "id": next(project._Project__artifacts_storage.keys()),
                "run_id": project.run_id,
                "key": "<key>",
                "date": nowstr,
                "learner": regression.estimator_name_,
                "dataset": joblib.hash(regression.y_test),
                "ml_task": regression._ml_task,
                "rmse": float(hash("<rmse_test>")),
                "log_loss": None,
                "roc_auc": None,
                "fit_time": float(hash("<fit_time>")),
                "predict_time": float(hash("<predict_time_test>")),
            },
            {
                "id": next(project._Project__artifacts_storage.keys()),
                "run_id": project.run_id,
                "key": "<key>",
                "date": nowstr,
                "learner": regression.estimator_name_,
                "dataset": joblib.hash(regression.y_test),
                "ml_task": regression._ml_task,
                "rmse": float(hash("<rmse_test>")),
                "log_loss": None,
                "roc_auc": None,
                "fit_time": float(hash("<fit_time>")),
                "predict_time": float(hash("<predict_time_test>")),
            },
        ]

    def test_delete(self, tmp_path, binary_classification, regression):
        project1 = Project("<project1>", workspace=tmp_path)
        project1.put("<project1-key1>", binary_classification)
        project1.put("<project1-key2>", regression)

        project2 = Project("<project2>", workspace=tmp_path)
        project2.put("<project2-key1>", binary_classification)

        assert len(DiskCacheStorage(tmp_path / "metadata")) == 3
        assert len(DiskCacheStorage(tmp_path / "artifacts")) == 2

        Project.delete("<project1>", workspace=tmp_path)

        assert len(DiskCacheStorage(tmp_path / "metadata")) == 1
        assert len(DiskCacheStorage(tmp_path / "artifacts")) == 1

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

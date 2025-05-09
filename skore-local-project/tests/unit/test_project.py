from io import BytesIO
from types import SimpleNamespace

import joblib
from pytest import fixture, raises
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore_local_project import Project
from skore_local_project.storage import DiskCacheStorage


class TestProject:
    @fixture
    def regression(self):
        X, y = make_regression(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return EstimatorReport(
            LinearRegression(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

    @fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, Datetime):
        monkeypatch.setattr("skore_local_project.project.datetime", Datetime)

    @fixture(autouse=True)
    def monkeypatch_metrics(self, monkeypatch, Datetime):
        monkeypatch.setattr(
            "skore.sklearn._estimator.metrics_accessor._MetricsAccessor.rmse",
            lambda _, data_source: float(hash(f"<rmse_{data_source}>")),
        )
        monkeypatch.setattr(
            "skore.sklearn._estimator.metrics_accessor._MetricsAccessor.timings",
            lambda self: {
                "fit_time": float(hash("<fit_time>")),
                "predict_time_test": float(hash("<predict_time_test>")),
                "predict_time_train": float(hash("<predict_time_train>")),
            },
        )

    def test_init(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "skore_local_project.project.platformdirs.user_cache_dir",
            lambda: str(tmp_path),
        )

        project = Project("<project>")

        assert project.workspace == str(tmp_path / "skore")
        assert project.name == "<project>"
        assert project.run_id
        assert isinstance(project.metadata_storage, DiskCacheStorage)
        assert isinstance(project.artifacts_storage, DiskCacheStorage)

    def test_pickle(self, regression):
        # Pickle the report once, without any value in the cache
        assert not regression._cache
        pickle_1 = Project.pickle(regression)

        # Pickle the same report, but with values in the cache
        regression.cache_predictions()
        assert regression._cache
        pickle_2 = Project.pickle(regression)

        # Make sure that the two pickles on the report are not affected by the cache
        assert pickle_1 == pickle_2

    def test_init_with_envar(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SKORE_WORKSPACE", str(tmp_path))
        project = Project("<project>")

        assert project.workspace == str(tmp_path)
        assert project.name == "<project>"
        assert project.run_id
        assert isinstance(project.metadata_storage, DiskCacheStorage)
        assert isinstance(project.artifacts_storage, DiskCacheStorage)

    def test_init_with_workspace(self, tmp_path):
        project = Project("<project>", workspace=tmp_path)

        assert project.workspace == str(tmp_path)
        assert project.name == "<project>"
        assert project.run_id
        assert isinstance(project.metadata_storage, DiskCacheStorage)
        assert isinstance(project.artifacts_storage, DiskCacheStorage)

    def test_put_exception(self, tmp_path):
        project = Project("<project>", workspace=tmp_path)

        with raises(TypeError, match="Key must be a string"):
            project.put(None, "<value>")

        with raises(TypeError, match="Report must be a `skore.EstimatorReport`"):
            project.put("<key>", "<value>")

    def test_put(self, tmp_path, nowstr, regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", regression)

        # Ensure artifacts are persisted
        assert len(project.artifacts_storage) == 1

        with BytesIO(next(project.artifacts_storage.values())) as stream:
            artifact = joblib.load(stream)

        assert isinstance(artifact, EstimatorReport)
        assert artifact.estimator_name_ == regression.estimator_name_
        assert artifact._ml_task == regression._ml_task

        # Ensure metadata are persisted
        assert len(project.metadata_storage) == 1
        assert list(project.metadata_storage.values()) == [
            {
                "project_name": "<project>",
                "run_id": project.run_id,
                "key": "<key>",
                "artifact_id": next(project.artifacts_storage.keys()),
                "date": nowstr,
                "learner": regression.estimator_name_,
                "dataset": joblib.hash(regression.y_test),
                "ml_task": regression._ml_task,
                "rmse": float(hash("<rmse_test>")),
                "log_loss": None,
                "roc_auc": None,
                "fit_time": float(hash("<fit_time>")),
                "predict_time": float(hash("<predict_time_test>")),
            }
        ]

    def test_reports(self, tmp_path):
        project = Project("<project>", workspace=tmp_path)

        assert isinstance(project.reports, SimpleNamespace)
        assert hasattr(project.reports, "get")
        assert hasattr(project.reports, "metadata")

    def test_reports_get(self, tmp_path, regression):
        project = Project("<project>", workspace=tmp_path)
        project.put("<key>", regression)
        project.put("<key>", regression)

        report = project.reports.get(next(project.artifacts_storage.keys()))

        assert len(project.artifacts_storage) == 1
        assert len(project.metadata_storage) == 2
        assert isinstance(report, EstimatorReport)
        assert report.estimator_name_ == regression.estimator_name_
        assert report._ml_task == regression._ml_task

    def test_reports_metadata(self, tmp_path, nowstr, regression):
        project = Project("<project>", workspace=tmp_path)

        project.put("<key>", regression)
        project.put("<key>", regression)

        assert len(project.artifacts_storage) == 1
        assert len(project.metadata_storage) == 2
        assert project.reports.metadata() == [
            {
                "id": next(project.artifacts_storage.keys()),
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
                "id": next(project.artifacts_storage.keys()),
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

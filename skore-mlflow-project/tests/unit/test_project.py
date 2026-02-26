from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
from sklearn.linear_model import LinearRegression

from skore_mlflow_project import Project
from skore_mlflow_project import _matplotlib as matplotlib_module
from skore_mlflow_project import project as project_module
from skore_mlflow_project.project import (
    _log_artifact,
    format_date,
    report_type,
)


def test_format_date() -> None:
    assert format_date(0) == "1970-01-01T00:00:00+00:00"
    assert format_date(None) == ""


def test_report_type_invalid_report() -> None:
    with pytest.raises(TypeError, match="Report must be a `skore.EstimatorReport`"):
        report_type(object())


def test_project_put_requires_string_key(tmp_path, regression) -> None:
    project = Project("<project>", tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    with pytest.raises(TypeError, match="Key must be a string"):
        project.put(1, regression)


def test_project_put_requires_supported_report_type(tmp_path) -> None:
    project = Project("<project>", tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    with pytest.raises(TypeError, match="Report must be a `skore.EstimatorReport`"):
        project.put("<key>", object())


def test_run_to_metadata_unsupported_report_type_raises() -> None:
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="id", run_name="name", start_time=0),
        data=SimpleNamespace(
            tags={
                "report_type": "unsupported",
                "learner": "L",
                "ml_task": "regression",
            },
            metrics={"fit_time": 0.1, "predict_time": 0.2},
        ),
    )

    with pytest.raises(ValueError, match="Unsupported report type: unsupported"):
        Project._run_to_metadata(run)


def test_log_model_falls_back_for_mlflow_2(monkeypatch) -> None:
    calls = []

    def _log_model(*args, **kwargs):
        calls.append(kwargs)
        if "name" in kwargs:
            raise TypeError("unexpected keyword argument 'name'")

    monkeypatch.setattr(mlflow.sklearn, "log_model", _log_model)

    project_module._log_model(LinearRegression(), input_example=None)

    assert len(calls) == 2


def test_log_model_reraises_unexpected_typeerror(monkeypatch) -> None:
    def _log_model(*args, **kwargs):
        raise TypeError("unexpected failure")

    monkeypatch.setattr(mlflow.sklearn, "log_model", _log_model)

    with pytest.raises(TypeError, match="unexpected failure"):
        project_module._log_model(LinearRegression(), input_example=None)


def test_log_artifact_raises_on_unsupported_payload() -> None:
    with pytest.raises(TypeError, match="Unexpected artifact payload type"):
        _log_artifact(project_module.Artifact("bad", 123))


def test_switch_mpl_backend_falls_back_when_restore_fails(monkeypatch) -> None:
    get_backend_values = iter(["TkAgg", "agg"])
    switch_calls = []

    monkeypatch.setattr(
        matplotlib_module.plt, "get_backend", lambda: next(get_backend_values)
    )

    def _switch_backend(backend):
        switch_calls.append(backend)
        if backend == "TkAgg":
            raise RuntimeError("restore failed")

    monkeypatch.setattr(matplotlib_module.plt, "switch_backend", _switch_backend)

    with matplotlib_module.switch_mpl_backend("agg"):
        pass

    assert switch_calls == ["agg", "TkAgg", "agg"]


class TestProject:
    CACHE_SENTINEL = ("__cache_sentinel__",)

    @staticmethod
    def tracking_uri(tmp_path):
        return f"sqlite:///{tmp_path}/mlflow.db"

    def test_init(self, tmp_path):
        tracking_uri = self.tracking_uri(tmp_path)

        project = Project("<project>", tracking_uri=tracking_uri)

        assert project.name == "<project>"
        assert project.tracking_uri == tracking_uri
        assert repr(project) == (
            f"Project(mode='mlflow', name='<project>', tracking_uri='{tracking_uri}')"
        )

    def test_put(self, tmp_path, regression):
        mlflow.set_tracking_uri(self.tracking_uri(tmp_path))
        project = Project("<project>")
        project.put("<key>", regression)

        summary = project.summarize()
        assert len(summary) == 1
        assert summary[0]["id"]
        assert summary[0]["key"] == "<key>"
        assert summary[0]["learner"] == "Ridge"
        assert summary[0]["ml_task"] == "regression"
        assert summary[0]["report_type"] == "estimator"
        assert summary[0]["dataset"]
        assert summary[0]["rmse"] is not None
        assert summary[0]["fit_time"] is not None

        run = mlflow.get_run(summary[0]["id"])
        assert "random_state" in run.data.params
        assert run.data.params["random_state"] == "42"
        assert run.data.metrics["rmse"] == pytest.approx(summary[0]["rmse"])
        assert "fit_time" in run.data.metrics

        report_dir = Path(
            mlflow.artifacts.download_artifacts(
                run_id=summary[0]["id"],
                tracking_uri=project.tracking_uri,
            )
        )
        assert (report_dir / "report.pkl").exists()
        assert (report_dir / "all_metrics.csv").exists()
        assert (report_dir / "metrics_details" / "prediction_error.csv").exists()
        assert (report_dir / "prediction_error.png").exists()
        assert (report_dir / "data.analyze.html").exists()

    def test_get(self, tmp_path, regression):
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))
        project.put("<key>", regression)

        summary = project.summarize()
        report = project.get(summary[0]["id"])
        predictions = report.estimator_.predict(regression.X_test)
        expected_predictions = regression.estimator_.predict(regression.X_test)

        assert len(predictions) == len(regression.X_test)
        assert predictions == pytest.approx(expected_predictions)

    def test_put_cross_validation(self, tmp_path, regression_cv):
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))
        project.put("<key>", regression_cv)

        summary = project.summarize()
        assert len(summary) == 1
        assert summary[0]["report_type"] == "cross-validation"
        assert summary[0]["ml_task"] == "multiclass-classification"
        assert summary[0]["dataset"]
        assert summary[0]["roc_auc_mean"] is not None
        assert summary[0]["fit_time_mean"] is not None

        run = mlflow.get_run(summary[0]["id"])
        assert run.data.metrics["roc_auc"] == pytest.approx(summary[0]["roc_auc_mean"])
        assert "random_state" in run.data.params
        assert run.data.params["random_state"] == "42"

        report_dir = Path(
            mlflow.artifacts.download_artifacts(
                run_id=summary[0]["id"],
                tracking_uri=project.tracking_uri,
            )
        )
        assert (report_dir / "report.pkl").exists()
        assert (report_dir / "all_metrics.csv").exists()
        assert (report_dir / "metrics_details" / "confusion_matrix.csv").exists()
        assert (report_dir / "confusion_matrix.png").exists()
        assert (report_dir / "metrics_details" / "roc.csv").exists()
        assert (report_dir / "roc.png").exists()
        assert (report_dir / "metrics_details" / "precision_recall.csv").exists()
        assert (report_dir / "precision_recall.png").exists()
        assert (report_dir / "timings.csv").exists()
        assert (report_dir / "metrics_details" / "per_split.csv").exists()

    def test_put_pickles_estimator_without_cache(self, tmp_path, regression):
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))
        original_cache = regression._cache
        regression._cache[self.CACHE_SENTINEL] = "present"

        project.put("<key>", regression)

        assert regression._cache is original_cache
        assert regression._cache[self.CACHE_SENTINEL] == "present"

        summary = project.summarize()
        restored = project.get(summary[0]["id"])
        assert self.CACHE_SENTINEL not in restored._cache

    def test_put_pickles_cv_without_cache(self, tmp_path, regression_cv):
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))
        root_cache = regression_cv._cache
        split_cache = regression_cv.estimator_reports_[0]._cache
        regression_cv._cache[self.CACHE_SENTINEL] = "root"
        regression_cv.estimator_reports_[0]._cache[self.CACHE_SENTINEL] = "split"

        project.put("<key>", regression_cv)

        assert regression_cv._cache is root_cache
        assert regression_cv.estimator_reports_[0]._cache is split_cache
        assert regression_cv._cache[self.CACHE_SENTINEL] == "root"
        assert (
            regression_cv.estimator_reports_[0]._cache[self.CACHE_SENTINEL] == "split"
        )

        summary = project.summarize()
        restored = project.get(summary[0]["id"])
        assert self.CACHE_SENTINEL not in restored._cache
        assert self.CACHE_SENTINEL not in restored.estimator_reports_[0]._cache

    def test_get_unknown_id(self, tmp_path):
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))

        with pytest.raises(KeyError):
            project.get("missing-run-id")

    def test_delete(self, tmp_path):
        tracking_uri = self.tracking_uri(tmp_path)
        project = Project("project", tracking_uri=tracking_uri)
        with pytest.raises(NotImplementedError):
            Project.delete(name=project.name)

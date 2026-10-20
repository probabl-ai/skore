import logging
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest
import skrub
from mlflow.exceptions import MlflowException
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from skore import CrossValidationReport
from skore._plugins.mlflow import Project
from skore._plugins.mlflow import project as project_module
from skore._plugins.mlflow.project import (
    _log_artifact,
    _storage_experiment_name,
    format_date,
    report_type,
)


def _run_id(report_id: str) -> str:
    runs = mlflow.search_runs(
        filter_string=f"tags.skore_report_id = '{report_id}'",
        output_format="list",
        max_results=1,
    )
    return runs[0].info.run_id


def test_format_date() -> None:
    assert format_date(0) == "1970-01-01T00:00:00+00:00"
    assert format_date(None) == ""


def test_storage_experiment_name_outside_databricks() -> None:
    assert _storage_experiment_name("sqlite:///mlflow.db") == "__skore-storage__"


@pytest.mark.parametrize("tracking_uri", ["databricks", "databricks://profile"])
def test_storage_experiment_name_on_databricks(monkeypatch, tracking_uri) -> None:
    request = {}

    def fake_http_request_safe(host_creds, endpoint, method):
        request.update(host_creds=host_creds, endpoint=endpoint, method=method)
        return SimpleNamespace(json=lambda: {"userName": "user@example.com"})

    monkeypatch.setattr(
        project_module, "get_databricks_host_creds", lambda uri: f"<creds:{uri}>"
    )
    monkeypatch.setattr(project_module, "http_request_safe", fake_http_request_safe)

    assert (
        _storage_experiment_name(tracking_uri)
        == "/Users/user@example.com/__skore-storage__"
    )
    assert request == {
        "host_creds": f"<creds:{tracking_uri}>",
        "endpoint": "/api/2.0/preview/scim/v2/Me",
        "method": "GET",
    }


def test_storage_experiment_name_reports_identity_lookup_failure(monkeypatch) -> None:
    def fake_http_request_safe(host_creds, endpoint, method):
        raise MlflowException("PERMISSION_DENIED")

    monkeypatch.setattr(project_module, "get_databricks_host_creds", lambda uri: None)
    monkeypatch.setattr(project_module, "http_request_safe", fake_http_request_safe)

    with pytest.raises(
        MlflowException, match="Failed to resolve the Databricks workspace user"
    ):
        _storage_experiment_name("databricks")


def test_report_type_invalid_report() -> None:
    with pytest.raises(TypeError, match="Report must be a `skore.EstimatorReport`"):
        report_type(object())


def test_project_put_requires_string_key(reg_report) -> None:
    project = Project("<project>")
    with pytest.raises(TypeError, match="Key must be a string"):
        project.put(1, reg_report)


def test_project_put_requires_supported_report_type() -> None:
    project = Project("<project>")
    with pytest.raises(TypeError, match="Report must be a `skore.EstimatorReport`"):
        project.put("<key>", object())


def test_project_put_raises_inside_active_run(reg_report) -> None:
    project = Project("<project>")

    with (
        mlflow.start_run(experiment_id=project.experiment_id),
        pytest.raises(RuntimeError, match=r"active run is already open"),
    ):
        project.put("<key>", reg_report)


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

    project_module._log_model(LinearRegression(), input_example=None, name="test")

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


def test_log_artifact_logs_series_metrics_as_csv(monkeypatch) -> None:
    logged: list[tuple[str, str]] = []

    def fake_log_text(text: str, artifact_file: str) -> None:
        logged.append((text, artifact_file))

    monkeypatch.setattr(mlflow, "log_text", fake_log_text)

    metrics = pd.Series({"accuracy": 0.9, "rmse": 1.2}, name="test")
    _log_artifact(project_module.Artifact("metrics", metrics))

    assert len(logged) == 1
    csv_text, artifact_file = logged[0]
    assert artifact_file == "metrics.csv"
    assert "accuracy" in csv_text
    assert "0.9" in csv_text


class TestProject:
    CLF_ARTIFACTS = [
        "metrics.confusion_matrix.png",
        "metrics_details/confusion_matrix.csv",
        "metrics_details/precision_recall.csv",
        "metrics_details/roc.csv",
        "metrics.precision_recall.png",
        "metrics.roc.png",
    ]

    def test_init_with_explicit_tracking_uri(self, mlflow_tracking_uri):
        tracking_uri = mlflow_tracking_uri()

        project = Project("<project>", tracking_uri=tracking_uri)

        assert project.name == "<project>"
        assert project.tracking_uri == tracking_uri
        assert repr(project) == (
            f"Project(name='<project>', mode='mlflow', tracking_uri='{tracking_uri}')"
        )

    def test_init_reuses_existing_storage_experiment(self, mlflow_tracking_uri):
        tracking_uri = mlflow_tracking_uri()

        first = Project("first", tracking_uri=tracking_uri)
        second = Project("second", tracking_uri=tracking_uri)

        assert (
            first._Project__storage_experiment_id
            == second._Project__storage_experiment_id
        )

    def test_init_stores_in_databricks_home_directory(
        self, monkeypatch, mlflow_tracking_uri
    ):
        monkeypatch.setattr(project_module, "is_databricks_uri", lambda uri: True)
        monkeypatch.setattr(
            project_module, "_databricks_user_name", lambda uri: "user@example.com"
        )

        project = Project("<project>", tracking_uri=mlflow_tracking_uri())
        storage_experiment = mlflow.get_experiment(
            project._Project__storage_experiment_id
        )

        assert storage_experiment.name == "/Users/user@example.com/__skore-storage__"

    def test_init_reuses_existing_project(self, mlflow_tracking_uri):
        tracking_uri = mlflow_tracking_uri()

        first = Project("<project>", tracking_uri=tracking_uri)
        second = Project("<project>", tracking_uri=tracking_uri)

        assert first.experiment_id == second.experiment_id
        assert (
            first._Project__storage_experiment_id
            == second._Project__storage_experiment_id
        )

    @pytest.mark.parametrize(
        ("report_fixture", "expected_ml_task", "expected_metric", "expected_artifacts"),
        [
            (
                "reg_report",
                "regression",
                "rmse",
                [
                    "metrics_details/prediction_error.csv",
                    "metrics.prediction_error.png",
                ],
            ),
            (
                "mreg_report",
                "multioutput-regression",
                "rmse",
                [],
            ),
            ("clf_report", "binary-classification", "roc_auc", CLF_ARTIFACTS),
            ("mclf_report", "multiclass-classification", "roc_auc", CLF_ARTIFACTS),
        ],
    )
    def test_put(
        self,
        request,
        report_fixture,
        expected_ml_task,
        expected_metric,
        expected_artifacts,
    ):
        report = request.getfixturevalue(report_fixture)
        project = Project("<project>")
        project.put("<key>", report)

        (metadata,) = project.summarize()
        assert metadata["report_type"] == "estimator"
        assert metadata["ml_task"] == expected_ml_task
        assert metadata["learner"] == report.estimator_name_
        assert metadata["dataset"]
        assert metadata[expected_metric] is not None
        assert metadata["fit_time"] is not None

        run = mlflow.get_run(_run_id(metadata["id"]))
        assert run.info.run_name == "<key>"

        report_dir = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                tracking_uri=project.tracking_uri,
            )
        )
        assert (report_dir / "report.pkl").exists()
        assert (report_dir / "metrics.csv").exists()
        assert (report_dir / "data.summarize.html").exists()
        for artifact in expected_artifacts:
            assert (report_dir / artifact).exists()

    def test_get(self, reg_report):
        project = Project("<project>")
        project.put("<key>", reg_report)

        summary = project.summarize()
        assert summary[0]["report_id"] == str(reg_report.id)
        report = project.get(summary[0]["id"])
        predictions = report.estimator_.predict(reg_report.X_test)
        expected_predictions = reg_report.estimator_.predict(reg_report.X_test)

        assert len(predictions) == len(reg_report.X_test)
        assert predictions == pytest.approx(expected_predictions)

    def test_storage_deduplicates_data_artifacts(self):
        project = Project("<project>")

        project._write_object_in_storage("<object-id>", b"payload")
        project._write_object_in_storage("<object-id>", b"payload")

        storage_runs = mlflow.search_runs(
            experiment_ids=[project._Project__storage_experiment_id],
            filter_string="attributes.run_name = '<object-id>'",
            output_format="list",
        )

        assert len(storage_runs) == 1
        artifact_path = project._Project__mlflow_client.download_artifacts(
            storage_runs[0].info.run_id, "obj"
        )
        assert Path(artifact_path).read_bytes() == b"payload"

    def test_put_cross_validation(self, cv_mclf_report):
        project = Project("<project>")
        project.put("<key>", cv_mclf_report)

        (metadata,) = project.summarize()
        assert metadata["report_type"] == "cross-validation"
        assert metadata["ml_task"] == "multiclass-classification"
        assert metadata["dataset"]
        assert metadata["roc_auc_mean"] is not None
        assert metadata["fit_time_mean"] is not None
        assert metadata["roc_auc_std"] is not None
        assert metadata["fit_time_std"] is not None
        assert metadata["predict_time_std"] is not None

        run = mlflow.get_run(_run_id(metadata["id"]))
        assert run.data.metrics["roc_auc"] == pytest.approx(metadata["roc_auc_mean"])
        assert run.data.metrics["roc_auc_std"] == pytest.approx(metadata["roc_auc_std"])
        assert run.data.metrics["fit_time_std"] == pytest.approx(
            metadata["fit_time_std"]
        )
        assert "random_state" in run.data.params
        assert run.data.params["random_state"] == "42"

        report_dir = Path(
            mlflow.artifacts.download_artifacts(
                run_id=_run_id(metadata["id"]),
                tracking_uri=project.tracking_uri,
            )
        )
        assert (report_dir / "report.pkl").exists()
        assert (report_dir / "metrics.csv").exists()
        assert (report_dir / "data.summarize.html").exists()
        for artifact in self.CLF_ARTIFACTS:
            assert (report_dir / artifact).exists()
        assert (report_dir / "metrics_details" / "per_split.csv").exists()

    def test_put_get_skrub_cross_validation(self) -> None:
        X, y = make_regression(n_samples=40, n_features=3, random_state=0)
        learner = (
            skrub.X(X).skb.apply(DummyRegressor(), y=skrub.y(y)).skb.make_learner()
        )
        report = CrossValidationReport(
            learner,
            data={"_skrub_X": X, "_skrub_y": y},
            splitter=KFold(n_splits=2),
        )
        project = Project("<project>")
        project.put("<key>", report)

        (metadata,) = project.summarize()
        run = mlflow.get_run(_run_id(metadata["id"]))
        assert run.data.tags["skore_status"] == "completed"

        restored = project.get(metadata["id"])
        assert isinstance(restored, CrossValidationReport)
        assert restored.metrics.rmse(aggregate="mean").iloc[0] == pytest.approx(
            report.metrics.rmse(aggregate="mean").iloc[0]
        )

    def test_get_unknown_id_with_explicit_tracking_uri(self, mlflow_tracking_uri):
        tracking_uri = mlflow_tracking_uri()
        project = Project("<project>", tracking_uri=tracking_uri)

        with pytest.raises(KeyError):
            project.get("missing-run-id")

    def test_delete(self, reg_report, mlflow_tracking_uri):
        tracking_uri = mlflow_tracking_uri()
        project = Project("project", tracking_uri=tracking_uri)
        project.put("<key>", reg_report)

        Project.delete(name=project.name, tracking_uri=tracking_uri)

        with pytest.raises(LookupError):
            Project.delete(name=project.name, tracking_uri=tracking_uri)


def test_databricks_user_name_requires_username(monkeypatch):
    """Service principals identified without ``userName`` are reported."""

    def fake_http_request_safe(host_creds, endpoint, method):
        return SimpleNamespace(json=lambda: {"applicationId": "service-principal"})

    monkeypatch.setattr(project_module, "get_databricks_host_creds", lambda uri: None)
    monkeypatch.setattr(project_module, "http_request_safe", fake_http_request_safe)

    with pytest.raises(MlflowException, match="no 'userName' field"):
        project_module._databricks_user_name("databricks")


def test_init_reraises_when_storage_experiment_is_missing(monkeypatch):
    """A failed storage-experiment lookup is not hidden by the reuse path."""

    def create_experiment(name):
        raise MlflowException("already exists")

    monkeypatch.setattr(mlflow, "create_experiment", create_experiment)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lambda name: None)

    with pytest.raises(MlflowException, match="already exists"):
        Project("<project>")


def test_log_iter_rejects_unknown_item():
    """The logger only accepts the declared MLflow payload types."""
    project = Project("<project>")

    with pytest.raises(TypeError, match="object"):
        project._log_iter([object()])


def test_get_rebuilds_cross_validation_report(monkeypatch):
    """A stored cross-validation payload is rebuilt as a cross-validation report."""
    project = Project("<project>")
    state = {"metadata": {"report_type": "cross-validation"}}
    sentinel = object()

    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda **kwargs: "report.pkl"
    )
    monkeypatch.setattr(project_module.joblib, "load", lambda path: state)
    monkeypatch.setattr(project_module, "internalize", lambda loaded, loader: state)
    monkeypatch.setattr(
        project_module.CrossValidationReport,
        "from_dict",
        staticmethod(lambda loaded: sentinel),
    )

    assert project.get("run-id") is sentinel


def test_summarize_skips_incomplete_runs(monkeypatch):
    """Runs that cannot be converted to metadata are omitted from the summary."""
    project = Project("<project>")
    runs = [SimpleNamespace(name="bad"), SimpleNamespace(name="good")]

    def run_to_metadata(run):
        if run.name == "bad":
            raise KeyError("learner")
        return {"id": "good"}

    monkeypatch.setattr(mlflow, "search_runs", lambda **kwargs: runs)
    monkeypatch.setattr(Project, "_run_to_metadata", staticmethod(run_to_metadata))

    assert project.summarize() == [{"id": "good"}]


def test_delete_without_tracking_uri():
    """Deletion uses the active tracking URI when none is passed explicitly."""
    project = Project("ephemeral-project")

    Project.delete(name=project.name)

    with pytest.raises(LookupError):
        Project.delete(name=project.name)


def test_delete_rejects_active_runs(monkeypatch):
    """An experiment with a running MLflow run cannot be deleted."""
    experiment = SimpleNamespace(lifecycle_stage="active", experiment_id="exp")
    client = SimpleNamespace(get_experiment_by_name=lambda name: experiment)
    monkeypatch.setattr(project_module, "MlflowClient", lambda: client)
    monkeypatch.setattr(mlflow, "search_runs", lambda **kwargs: [SimpleNamespace()])

    with pytest.raises(RuntimeError, match="active runs are still in progress"):
        Project.delete(name="project")


def test_load_object_from_storage_missing():
    """Reading an object that was never stored raises ``FileNotFoundError``."""
    project = Project("<project>")

    with pytest.raises(FileNotFoundError):
        project._load_object_from_storage("missing-object")


def test_suppress_log_warning_drops_matching_records():
    """Only log records matching the configured pattern are filtered out."""
    records = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("skore.test.mlflow")
    logger.setLevel(logging.WARNING)
    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        with project_module._suppress_log_warning("skore.test.mlflow", r"pickle"):
            logger.warning("Saving scikit-learn models in the pickle format")
            logger.warning("kept")
    finally:
        logger.removeHandler(handler)

    assert records == ["kept"]


def test_log_artifact_logs_json_payloads(monkeypatch):
    """Lists and dictionaries are stored as JSON artifacts."""
    logged = []
    monkeypatch.setattr(
        mlflow,
        "log_dict",
        lambda payload, artifact_file: logged.append((payload, artifact_file)),
    )

    _log_artifact(project_module.Artifact("values", ["a", "b"]))
    _log_artifact(project_module.Artifact("mapping", {"a": 1}))

    assert logged == [
        ({"values": ["a", "b"]}, "values.json"),
        ({"a": 1}, "mapping.json"),
    ]


def test_log_figure_falls_back_when_save_kwargs_are_unsupported(monkeypatch):
    """Older MLflow builds save the figure locally before logging the artifact."""

    class _Figure:
        def savefig(self, path, **kwargs):
            path.write_bytes(b"png")

    logged = []

    def log_figure(*args, **kwargs):
        raise TypeError("unexpected keyword argument 'save_kwargs'")

    monkeypatch.setattr(mlflow, "log_figure", log_figure)
    monkeypatch.setattr(mlflow, "log_artifact", lambda **kwargs: logged.append(kwargs))

    project_module._log_figure(_Figure(), "nested/plot.png")
    project_module._log_figure(_Figure(), "plot.png")

    assert logged[0]["artifact_path"] == "nested"
    assert logged[1]["artifact_path"] is None


def test_log_figure_reraises_unexpected_typeerror(monkeypatch):
    """Figure logging only translates the missing ``save_kwargs`` failure."""

    def log_figure(*args, **kwargs):
        raise TypeError("unexpected failure")

    monkeypatch.setattr(mlflow, "log_figure", log_figure)

    with pytest.raises(TypeError, match="unexpected failure"):
        project_module._log_figure(object(), "plot.png")

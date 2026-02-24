from pathlib import Path

import mlflow
import pytest

from skore_mlflow_project import Project


class TestProject:
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
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))
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
                artifact_path="report",
                tracking_uri=project.tracking_uri,
            )
        )
        assert (report_dir / "report.pkl").exists()
        assert (report_dir / "all_metrics.csv").exists()
        assert (report_dir / "metrics_details" / "prediction_error.csv").exists()
        assert (report_dir / "prediction_error.png").exists()
        assert (report_dir / "timings.json").exists()
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
        assert summary[0]["rmse_mean"] is not None
        assert summary[0]["fit_time_mean"] is not None

        run = mlflow.get_run(summary[0]["id"])
        assert run.data.metrics["rmse"] == pytest.approx(summary[0]["rmse_mean"])
        assert "random_state" in run.data.params
        assert run.data.params["random_state"] == "42"

        report_dir = Path(
            mlflow.artifacts.download_artifacts(
                run_id=summary[0]["id"],
                artifact_path="report",
                tracking_uri=project.tracking_uri,
            )
        )
        assert (report_dir / "report.pkl").exists()
        assert (report_dir / "all_metrics.csv").exists()
        assert (report_dir / "metrics_details" / "prediction_error.csv").exists()
        assert (report_dir / "prediction_error.png").exists()
        assert (report_dir / "timings.csv").exists()
        assert (report_dir / "metrics_details" / "per_split.csv").exists()

    def test_get_unknown_id(self, tmp_path):
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))

        with pytest.raises(KeyError):
            project.get("missing-run-id")

    def test_delete(self, tmp_path):
        tracking_uri = self.tracking_uri(tmp_path)
        project = Project("project", tracking_uri=tracking_uri)
        with pytest.raises(NotImplementedError):
            Project.delete(name=project.name)

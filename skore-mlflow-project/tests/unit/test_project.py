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

    def test_get(self, tmp_path, regression):
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))
        project.put("<key>", regression)

        summary = project.summarize()
        report = project.get(summary[0]["id"])
        predictions = report.estimator_.predict(regression.X_test)
        expected_predictions = regression.estimator_.predict(regression.X_test)

        assert len(predictions) == len(regression.X_test)
        assert predictions == pytest.approx(expected_predictions)

    def test_get_unknown_id(self, tmp_path):
        project = Project("<project>", tracking_uri=self.tracking_uri(tmp_path))

        with pytest.raises(KeyError):
            project.get("missing-run-id")

    def test_delete(self, tmp_path):
        tracking_uri = self.tracking_uri(tmp_path)
        project = Project("project", tracking_uri=tracking_uri)
        with pytest.raises(NotImplementedError):
            Project.delete(name=project.name)

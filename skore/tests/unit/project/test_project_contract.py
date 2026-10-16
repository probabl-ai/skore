"""Parametric contract tests for the public ``skore.Project`` API."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from httpx import Response
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge

from skore import EstimatorReport, Project, evaluate
from skore._plugins.hub.client.client import HUBClient
from skore._project.summary import Summary


@pytest.fixture
def regression_report() -> EstimatorReport:
    X, y = make_regression(random_state=42)
    return evaluate(Ridge(random_state=42), X, y)


@pytest.fixture
def second_regression_report() -> EstimatorReport:
    X, y = make_regression(random_state=7)
    return evaluate(LinearRegression(), X, y)


class TestLocalProjectContract:
    def test_api_contract(self, tmp_path, regression_report, second_regression_report):
        project = Project(
            name="contract-local",
            mode="local",
            workspace=Path(tmp_path),
        )

        assert project.mode == "local"
        assert project.name == "contract-local"
        assert project.workspace == Path(tmp_path)
        assert project.tracking_uri is None

        project.put("first", regression_report)
        project.put("second", second_regression_report)

        summary = project.summarize()
        assert isinstance(summary, Summary)
        assert len(summary.frame()) == 2

        dates = summary.frame()["date"].tolist()
        assert dates == sorted(dates)

        for report_id in summary.frame().index.get_level_values("id"):
            retrieved = project.get(report_id)
            assert retrieved.ml_task == "regression"

        Project.delete(
            name="contract-local",
            mode="local",
            workspace=Path(tmp_path),
        )

        with pytest.raises(LookupError):
            Project.delete(
                name="contract-local",
                mode="local",
                workspace=Path(tmp_path),
            )


class TestMlflowProjectContract:
    @pytest.fixture(autouse=True)
    def isolated_mlflow_tracking(self, tmp_path, monkeypatch, mlflow_tracking_uri):
        monkeypatch.chdir(tmp_path)
        previous_tracking_uri = mlflow.get_tracking_uri()
        tracking_uri = mlflow_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        try:
            yield tracking_uri
        finally:
            while mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.set_tracking_uri(previous_tracking_uri)

    def test_api_contract(
        self, regression_report, second_regression_report, isolated_mlflow_tracking
    ):
        project = Project(
            name="contract-mlflow",
            mode="mlflow",
            tracking_uri=isolated_mlflow_tracking,
        )

        assert project.mode == "mlflow"
        assert project.name == "contract-mlflow"
        assert project.workspace is None
        assert project.tracking_uri == isolated_mlflow_tracking

        project.put("first", regression_report)
        project.put("second", second_regression_report)

        summary = project.summarize()
        assert len(summary.frame()) == 2

        dates = summary.frame()["date"].tolist()
        assert dates == sorted(dates)

        for report_id in summary.frame().index.get_level_values("id"):
            retrieved = project.get(report_id)
            assert retrieved.ml_task == "regression"

        Project.delete(
            name="contract-mlflow",
            mode="mlflow",
            tracking_uri=isolated_mlflow_tracking,
        )

        with pytest.raises(LookupError):
            Project.delete(
                name="contract-mlflow",
                mode="mlflow",
                tracking_uri=isolated_mlflow_tracking,
            )


@pytest.fixture
def monkeypatch_hub_client(monkeypatch):
    monkeypatch.setenv("SKORE_HUB_API_KEY", "<key>")
    monkeypatch.setitem(HUBClient.__init__.__kwdefaults__, "retry", False)


@pytest.mark.respx()
@pytest.mark.usefixtures("monkeypatch_hub_client")
class TestHubProjectContract:
    def test_api_contract(self, regression_report, respx_mock, monkeypatch):
        monkeypatch.setattr(
            "skore._plugins.hub.artifact.media.data.TableReport.content_to_upload",
            lambda self: None,
        )
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/contract-hub",
                Response(200, json={"id": 42, "url": "http://domain/x"}),
            ),
            (
                "post",
                "projects/workspace/contract-hub/artifacts",
                Response(200, json=[]),
            ),
            (
                "post",
                "projects/workspace/contract-hub/estimator-reports",
                Response(201, json={"id": 42}),
            ),
            (
                "get",
                "projects/workspace/contract-hub/reports",
                Response(200, json={"next_cursor": None, "items": []}),
            ),
            ("delete", "/projects/workspace/contract-hub", Response(204)),
        ]
        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(
            name="contract-hub",
            mode="hub",
            workspace="workspace",
            host="http://localhost",
        )

        assert project.mode == "hub"
        assert project.name == "contract-hub"
        assert project.workspace == "workspace"
        assert project.tracking_uri is None

        project.put("first", regression_report)

        summary = project.summarize()
        assert isinstance(summary, Summary)

        Project.delete(
            name="contract-hub",
            mode="hub",
            workspace="workspace",
            host="http://localhost",
        )

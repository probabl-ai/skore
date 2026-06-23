"""Parametric contract tests for the public ``skore.Project`` API."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from httpx import Response
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from skore import EstimatorReport, Project
from skore._project._summary import Summary


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
    def isolated_mlflow_tracking(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        previous_tracking_uri = mlflow.get_tracking_uri()
        tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
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


@pytest.mark.respx()
class TestHubProjectContract:
    @pytest.fixture(autouse=True)
    def hub_client(self, monkeypatch):
        from urllib.parse import urljoin

        from httpx import Client

        class FakeClient(Client):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def request(self, method, url, **kwargs):
                response = super().request(
                    method, urljoin("http://localhost", url), **kwargs
                )
                response.raise_for_status()
                return response

        monkeypatch.setattr("skore._plugins.hub.project.project.HUBClient", FakeClient)
        monkeypatch.setattr("skore._plugins.hub.artifact.upload.HUBClient", FakeClient)

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
                "projects/workspace/contract-hub/estimator-reports/",
                Response(200, json=[]),
            ),
            (
                "get",
                "projects/workspace/contract-hub/cross-validation-reports/",
                Response(200, json=[]),
            ),
            ("delete", "/projects/workspace/contract-hub", Response(204)),
        ]
        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(
            name="contract-hub",
            mode="hub",
            workspace="workspace",
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
        )

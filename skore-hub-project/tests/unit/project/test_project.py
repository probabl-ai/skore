from functools import partialmethod
from io import BytesIO
from json import dumps, loads
from types import SimpleNamespace
from urllib.parse import urljoin

import joblib
from httpx import Client, Response
from pytest import fixture, mark, raises
from skore import EstimatorReport
from skore_hub_project import Project
from skore_hub_project.report import (
    CrossValidationReportPayload,
    EstimatorReportPayload,
)


class FakeClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def request(self, method, url, **kwargs):
        response = super().request(method, urljoin("http://localhost", url), **kwargs)
        response.raise_for_status()

        return response


@fixture(autouse=True)
def monkeypatch_client(monkeypatch):
    monkeypatch.setattr(
        "skore_hub_project.project.project.HUBClient",
        FakeClient,
    )
    monkeypatch.setattr(
        "skore_hub_project.artefact.upload.HUBClient",
        FakeClient,
    )


@fixture(scope="module")
def regression():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from skore import EstimatorReport

    X, y = make_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture(autouse=True)
def monkeypatch_permutation(monkeypatch):
    import skore

    monkeypatch.setattr(
        "skore.EstimatorReport.feature_importance.permutation",
        partialmethod(
            skore.EstimatorReport.feature_importance.permutation,
            seed=42,
        ),
    )


@fixture(autouse=True)
def monkeypatch_to_json(monkeypatch):
    monkeypatch.setattr(
        "skore._sklearn._plot.TableReportDisplay._to_json", lambda self: "[0,1]"
    )


class TestProject:
    def test_tenant(self):
        assert Project("<tenant>", "<name>").tenant == "<tenant>"

    def test_name(self):
        assert Project("<tenant>", "<name>").name == "<name>"

    @mark.respx(assert_all_called=True)
    def test_run_id(self, respx_mock):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": 0})
        )

        assert Project("<tenant>", "<name>").run_id == 0

    def test_put_exception(self, respx_mock):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": 0})
        )

        with raises(TypeError, match="Key must be a string"):
            Project("<tenant>", "<name>").put(None, "<value>")

        with raises(
            TypeError,
            match="must be a `skore.EstimatorReport` or `skore.CrossValidationReport`",
        ):
            Project("<tenant>", "<name>").put("<key>", "<value>")

    def test_put_estimator_report(self, monkeypatch, binary_classification, respx_mock):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": 0})
        )
        respx_mock.post("projects/<tenant>/<name>/artefacts").mock(
            Response(200, json=[])
        )
        respx_mock.post("projects/<tenant>/<name>/estimator-reports").mock(
            Response(200)
        )

        project = Project("<tenant>", "<name>")
        project.put("<key>", binary_classification)

        # Retrieve the content of the request
        content = loads(respx_mock.calls.last.request.content.decode())
        desired = loads(
            dumps(
                EstimatorReportPayload(
                    project=project, key="<key>", report=binary_classification
                ).model_dump()
            )
        )

        # Compare content with the desired output
        assert content == desired

    def test_put_cross_validation_report(
        self, monkeypatch, small_cv_binary_classification, respx_mock
    ):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": 0})
        )
        respx_mock.post("projects/<tenant>/<name>/artefacts").mock(
            Response(200, json=[])
        )
        respx_mock.post("projects/<tenant>/<name>/cross-validation-reports").mock(
            Response(200)
        )

        project = Project("<tenant>", "<name>")
        project.put("<key>", small_cv_binary_classification)

        # Retrieve the content of the request
        content = loads(respx_mock.calls.last.request.content.decode())
        desired = loads(
            dumps(
                CrossValidationReportPayload(
                    project=project, key="<key>", report=small_cv_binary_classification
                ).model_dump()
            )
        )

        # Compare content with the desired output
        assert content == desired

    def test_reports(self, respx_mock):
        url = "projects/<tenant>/<name>/runs"
        respx_mock.post(url).mock(Response(200, json={"id": 0}))

        project = Project("<tenant>", "<name>")

        assert isinstance(project.reports, SimpleNamespace)
        assert hasattr(project.reports, "get")
        assert hasattr(project.reports, "metadata")

    def test_reports_get(self, respx_mock, regression):
        # Mock hub routes that will be called
        url = "projects/<tenant>/<name>/runs"
        response = Response(200, json={"id": 0})
        respx_mock.post(url).mock(response)

        url = "projects/<tenant>/<name>/experiments/estimator-reports/<report_id>"
        response = Response(200, json={"raw": {"checksum": "<checksum>"}})
        respx_mock.get(url).mock(response)

        url = "projects/<tenant>/<name>/artefacts/read"
        response = Response(200, json=[{"url": "http://url.com"}])
        respx_mock.get(url).mock(response)

        with BytesIO() as stream:
            joblib.dump(regression, stream)

            url = "http://url.com"
            response = Response(200, content=stream.getvalue())
            respx_mock.get(url).mock(response)

        # Test
        project = Project("<tenant>", "<name>")
        report = project.reports.get("<report_id>")

        assert isinstance(report, EstimatorReport)
        assert report.estimator_name_ == regression.estimator_name_
        assert report._ml_task == regression._ml_task

    def test_reports_metadata(self, nowstr, respx_mock):
        url = "projects/<tenant>/<name>/runs"
        respx_mock.post(url).mock(Response(200, json={"id": 2}))

        url = "projects/<tenant>/<name>/experiments/estimator-reports"
        respx_mock.get(url).mock(
            Response(
                200,
                json=[
                    {
                        "id": "<report_id_0>",
                        "run_id": 0,
                        "key": "<key>",
                        "ml_task": "<ml_task>",
                        "estimator_class_name": "<estimator_class_name>",
                        "dataset_fingerprint": "<dataset_fingerprint>",
                        "created_at": nowstr,
                        "metrics": [
                            {"name": "rmse", "value": 0, "data_source": "train"},
                            {"name": "rmse", "value": 1, "data_source": "test"},
                        ],
                    },
                    {
                        "id": "<report_id_1>",
                        "run_id": 1,
                        "key": "<key>",
                        "ml_task": "<ml_task>",
                        "estimator_class_name": "<estimator_class_name>",
                        "dataset_fingerprint": "<dataset_fingerprint>",
                        "created_at": nowstr,
                        "metrics": [
                            {"name": "log_loss", "value": 0, "data_source": "train"},
                            {"name": "log_loss", "value": 1, "data_source": "test"},
                        ],
                    },
                ],
            )
        )

        project = Project("<tenant>", "<name>")
        metadata = project.reports.metadata()

        assert metadata == [
            {
                "id": "<report_id_0>",
                "run_id": 0,
                "key": "<key>",
                "date": nowstr,
                "learner": "<estimator_class_name>",
                "dataset": "<dataset_fingerprint>",
                "ml_task": "<ml_task>",
                "rmse": 1,
                "log_loss": None,
                "roc_auc": None,
                "fit_time": None,
                "predict_time": None,
            },
            {
                "id": "<report_id_1>",
                "run_id": 1,
                "key": "<key>",
                "date": nowstr,
                "learner": "<estimator_class_name>",
                "dataset": "<dataset_fingerprint>",
                "ml_task": "<ml_task>",
                "rmse": None,
                "log_loss": 1,
                "roc_auc": None,
                "fit_time": None,
                "predict_time": None,
            },
        ]

    def test_delete(self, respx_mock):
        respx_mock.delete("projects/<tenant>/<name>").mock(Response(204))
        Project.delete("<tenant>", "<name>")

    def test_delete_exception(self, respx_mock):
        respx_mock.delete("projects/<tenant>/<name>").mock(Response(403))

        with raises(
            PermissionError,
            match="Failed to delete the project; please contact the '<tenant>' owner",
        ):
            Project.delete("<tenant>", "<name>")

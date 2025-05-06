from json import dumps
from types import SimpleNamespace
from urllib.parse import urljoin

from httpx import Client, Response
from pytest import fixture, mark, raises
from skore_remote_project import Project
from skore_remote_project.item.skore_estimator_report_item import (
    SkoreEstimatorReportItem,
)


class FakeClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def request(self, method, url, **kwargs):
        response = super().request(method, urljoin("http://localhost", url), **kwargs)
        response.raise_for_status()

        return response


class TestProject:
    @fixture(scope="class")
    def regression(self):
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
    def monkeypatch_client(self, monkeypatch):
        monkeypatch.setattr(
            "skore_remote_project.project.project.AuthenticatedClient",
            FakeClient,
        )

    def test_tenant(self):
        assert Project("<tenant>", "<name>").tenant == "<tenant>"

    def test_name(self):
        assert Project("<tenant>", "<name>").name == "<name>"

    @mark.respx(assert_all_called=True)
    def test_run_id(self, respx_mock):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": "<run_id>"})
        )

        assert Project("<tenant>", "<name>").run_id == "<run_id>"

    def test_put_exception(self):
        with raises(TypeError, match="Key must be a string"):
            Project("<tenant>", "<name>").put(None, "<value>")

        with raises(TypeError, match="Note must be a string"):
            Project("<tenant>", "<name>").put("<key>", "<value>", note=0)

    def test_put(self, respx_mock):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": "<run_id>"})
        )
        respx_mock.post("projects/<tenant>/<name>/items").mock(Response(200))

        Project("<tenant>", "<name>").put("<key>", "<value>", note="<note>")

        assert respx_mock.calls.last.request.content == str.encode(
            dumps(
                {
                    "representation": {
                        "media_type": "application/json",
                        "value": "<value>",
                    },
                    "parameters": {
                        "class": "JSONableItem",
                        "parameters": {"value": "<value>"},
                    },
                    "key": "<key>",
                    "run_id": "<run_id>",
                    "note": "<note>",
                },
                separators=(",", ":"),
            )
        )

    def test_experiments(self, respx_mock):
        url = "projects/<tenant>/<name>/runs"
        respx_mock.post(url).mock(Response(200, json={"id": "<run_id>"}))

        project = Project("<tenant>", "<name>")

        assert isinstance(project.experiments, SimpleNamespace)
        assert hasattr(project.experiments, "get")
        assert hasattr(project.experiments, "metadata")

    def test_experiments_get(self, respx_mock, regression):
        from skore import EstimatorReport

        url = "projects/<tenant>/<name>/runs"
        respx_mock.post(url).mock(Response(200, json={"id": "<run_id>"}))

        url = "projects/<tenant>/<name>/experiments/estimator-reports/<experiment_id>"
        respx_mock.get(url).mock(
            Response(
                200,
                json={
                    "raw": {
                        "class": "SkoreEstimatorReportItem",
                        "parameters": {
                            "pickle_b64_str": (
                                SkoreEstimatorReportItem.factory(
                                    regression
                                ).pickle_b64_str
                            )
                        },
                    }
                },
            )
        )

        project = Project("<tenant>", "<name>")
        experiment = project.experiments.get("<experiment_id>")

        assert isinstance(experiment, EstimatorReport)
        assert experiment.estimator_name_ == regression.estimator_name_
        assert experiment._ml_task == regression._ml_task

    def test_experiments_metadata(self, nowstr, respx_mock):
        url = "projects/<tenant>/<name>/runs"
        respx_mock.post(url).mock(Response(200, json={"id": "<run_id_2>"}))

        url = "projects/<tenant>/<name>/experiments/estimator-reports"
        respx_mock.get(url).mock(
            Response(
                200,
                json=[
                    {
                        "id": "<experiment_id_0>",
                        "run_id": "<run_id_0>",
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
                        "id": "<experiment_id_1>",
                        "run_id": "<run_id_1>",
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
        metadata = project.experiments.metadata()

        assert metadata == [
            {
                "id": "<experiment_id_0>",
                "run_id": "<run_id_0>",
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
                "id": "<experiment_id_1>",
                "run_id": "<run_id_1>",
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

from io import BytesIO
from json import loads
from math import ceil
from operator import itemgetter
from types import SimpleNamespace
from urllib.parse import urljoin

import joblib
from httpx import Client, Response
from pytest import fixture, mark, raises
from skore import EstimatorReport
from skore_hub_project import Project
from skore_hub_project.artefact.serializer import Serializer


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

        with raises(TypeError, match="Report must be a `skore.EstimatorReport`"):
            Project("<tenant>", "<name>").put("<key>", "<value>")

    def test_upload_in_put(self, monkeypatch, respx_mock, regression):
        cache = regression._cache
        regression._cache = {}

        try:
            with Serializer(regression) as serializer:
                pickle = serializer.filepath.read_bytes()
                checksum = serializer.checksum
                chunk_size = ceil(len(pickle) / 2)
        finally:
            regression._cache = cache

        monkeypatch.setattr("skore_hub_project.artefact.upload.CHUNK_SIZE", chunk_size)
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": 0})
        )
        respx_mock.post("projects/<tenant>/<name>/artefacts").mock(
            Response(
                200,
                json=[
                    {"upload_url": "http://chunk2.com/", "chunk_id": 2},
                    {"upload_url": "http://chunk1.com/", "chunk_id": 1},
                ],
            )
        )
        respx_mock.put("http://chunk1.com").mock(
            Response(200, headers={"etag": '"<etag1>"'})
        )
        respx_mock.put("http://chunk2.com").mock(
            Response(200, headers={"etag": '"<etag2>"'})
        )
        respx_mock.post("projects/<tenant>/<name>/artefacts/complete")
        respx_mock.post("projects/<tenant>/<name>/estimator-reports")

        Project("<tenant>", "<name>").put("<key>", regression)

        requests = [call.request for call in respx_mock.calls]

        assert len(requests) == 6
        assert requests[0].url.path == "/projects/<tenant>/<name>/runs"
        assert requests[1].url.path == "/projects/<tenant>/<name>/artefacts"
        assert loads(requests[1].content.decode()) == [
            {
                "checksum": checksum,
                "chunk_number": 2,
                "content_type": "estimator-report",
            }
        ]
        assert sorted(
            (
                (str(requests[2].url), requests[2].content),
                (str(requests[3].url), requests[3].content),
            ),
            key=itemgetter(0),
        ) == [
            ("http://chunk1.com/", pickle[:chunk_size]),
            ("http://chunk2.com/", pickle[chunk_size:]),
        ]
        assert requests[4].url.path == "/projects/<tenant>/<name>/artefacts/complete"
        assert loads(requests[4].content.decode()) == [
            {
                "checksum": checksum,
                "etags": {
                    "1": '"<etag1>"',
                    "2": '"<etag2>"',
                },
            }
        ]

    def test_put(self, respx_mock, regression):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": 0})
        )
        respx_mock.post("projects/<tenant>/<name>/artefacts").mock(
            Response(200, json=[])
        )
        respx_mock.post("projects/<tenant>/<name>/estimator-reports").mock(
            Response(200)
        )

        Project("<tenant>", "<name>").put("<key>", regression)

        cache = regression._cache
        regression._cache = {}

        try:
            with Serializer(regression) as serializer:
                checksum = serializer.checksum
        finally:
            regression._cache = cache

        # Retrieve the content of the request
        content = loads(respx_mock.calls.last.request.content.decode())

        # Prepare the content to be compared
        content["dataset_fingerprint"] = None

        for item in content["related_items"]:
            item["representation"]["value"] = None

        for metric in content["metrics"]:
            metric["value"] = None

        # Compare content with the desired output
        assert content == {
            "dataset_fingerprint": None,
            "estimator_class_name": "LinearRegression",
            "metrics": [
                {
                    "name": "r2",
                    "verbose_name": "R²",
                    "value": None,
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
                },
                {
                    "name": "r2",
                    "verbose_name": "R²",
                    "value": None,
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                },
                {
                    "name": "rmse",
                    "verbose_name": "RMSE",
                    "value": None,
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": 3,
                },
                {
                    "name": "rmse",
                    "verbose_name": "RMSE",
                    "value": None,
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": 3,
                },
                {
                    "name": "fit_time",
                    "verbose_name": "Fit time (s)",
                    "value": None,
                    "data_source": None,
                    "greater_is_better": False,
                    "position": 1,
                },
                {
                    "name": "predict_time",
                    "verbose_name": "Predict time (s)",
                    "value": None,
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": 2,
                },
                {
                    "name": "predict_time",
                    "verbose_name": "Predict time (s)",
                    "value": None,
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": 2,
                },
            ],
            "ml_task": "regression",
            "related_items": [
                {
                    "key": "coefficients",
                    "verbose_name": "Feature importance - Coefficients",
                    "category": "feature_importance",
                    "attributes": {"method": "coefficients"},
                    "parameters": {},
                    "representation": {
                        "media_type": "application/vnd.dataframe",
                        "value": None,
                    },
                },
                {
                    "key": "estimator_html_repr",
                    "verbose_name": "Estimator HTML representation",
                    "category": "model",
                    "attributes": {},
                    "parameters": {},
                    "representation": {"media_type": "text/html", "value": None},
                },
                {
                    "key": "permutation",
                    "verbose_name": "Feature importance - Permutation",
                    "category": "feature_importance",
                    "attributes": {"data_source": "test", "method": "permutation"},
                    "parameters": {},
                    "representation": {
                        "media_type": "application/vnd.dataframe",
                        "value": None,
                    },
                },
                {
                    "key": "permutation",
                    "verbose_name": "Feature importance - Permutation",
                    "category": "feature_importance",
                    "attributes": {"data_source": "train", "method": "permutation"},
                    "parameters": {},
                    "representation": {
                        "media_type": "application/vnd.dataframe",
                        "value": None,
                    },
                },
                {
                    "key": "prediction_error",
                    "verbose_name": "Prediction error",
                    "category": "performance",
                    "attributes": {"data_source": "test"},
                    "parameters": {},
                    "representation": {
                        "media_type": "image/svg+xml;base64",
                        "value": None,
                    },
                },
                {
                    "key": "prediction_error",
                    "verbose_name": "Prediction error",
                    "category": "performance",
                    "attributes": {"data_source": "train"},
                    "parameters": {},
                    "representation": {
                        "media_type": "image/svg+xml;base64",
                        "value": None,
                    },
                },
                {
                    "key": "table_report",
                    "verbose_name": "Table report",
                    "category": "data",
                    "attributes": {"data_source": "test"},
                    "parameters": {},
                    "representation": {
                        "media_type": "application/vnd.skrub.table-report.v1",
                        "value": None,
                    },
                },
                {
                    "key": "table_report",
                    "verbose_name": "Table report",
                    "category": "data",
                    "attributes": {"data_source": "train"},
                    "parameters": {},
                    "representation": {
                        "media_type": "application/vnd.skrub.table-report.v1",
                        "value": None,
                    },
                },
            ],
            "parameters": {"checksum": checksum},
            "key": "<key>",
            "run_id": 0,
        }

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

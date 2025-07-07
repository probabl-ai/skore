import io
from json import loads
from types import SimpleNamespace
from urllib.parse import urljoin

import joblib
from httpx import Client, Response
from pytest import fixture, mark, raises
from skore import EstimatorReport
from skore_hub_project import Project
from skore_hub_project.project.project import dumps


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
            "skore_hub_project.project.project.AuthenticatedClient",
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

        with raises(TypeError, match="Report must be a `skore.EstimatorReport`"):
            Project("<tenant>", "<name>").put("<key>", "<value>")

    def test_put_with_upload(self, respx_mock, regression):
        pickle, checksum = dumps(regression)
        artefacts_response = Response(200, json=[{"upload_url": "http://s3.com"}])
        runs_response = Response(200, json={"id": "<run_id>"})

        respx_mock.post("projects/<tenant>/<name>/artefacts").mock(artefacts_response)
        respx_mock.put("http://s3.com")
        respx_mock.post("projects/<tenant>/<name>/artefacts/complete")
        respx_mock.post("projects/<tenant>/<name>/runs").mock(runs_response)
        respx_mock.post("projects/<tenant>/<name>/items")

        Project("<tenant>", "<name>").put("<key>", regression)

        # Ensure what is sent to artefacts route
        artefacts_request = respx_mock.calls[0].request

        assert artefacts_request.url.path == "/projects/<tenant>/<name>/artefacts"
        assert loads(artefacts_request.content.decode()) == [
            {
                "checksum": checksum,
                "content_type": "estimator-report-pickle",
            }
        ]

        # Ensure what is sent to S3 route
        S3_request = respx_mock.calls[1].request

        assert S3_request.url == "http://s3.com/"
        assert S3_request.content == pickle

        # Ensure what is sent to complete route
        complete_request = respx_mock.calls[2].request

        assert (
            complete_request.url.path == "/projects/<tenant>/<name>/artefacts/complete"
        )
        assert loads(complete_request.content.decode()) == [
            {"checksum": checksum, "etags": {}}
        ]

        # Ensure what is sent to items route
        content = loads(respx_mock.calls.last.request.content.decode())

        ## Prepare the content to be compared
        content["dataset_fingerprint"] = None

        for item in content["related_items"]:
            item["representation"]["value"] = None

        for metric in content["metrics"]:
            metric["value"] = None

        ## Compare content with the desired output
        assert content == {
            "dataset_fingerprint": None,
            "estimator_class_name": "LinearRegression",
            "estimator_hyper_params": {},
            "metrics": [
                {
                    "name": "r2",
                    "verbose_name": "R²",
                    "value": None,
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                },
                {
                    "name": "r2",
                    "verbose_name": "R²",
                    "value": None,
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
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
                    "name": "rmse",
                    "verbose_name": "RMSE",
                    "value": None,
                    "data_source": "test",
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
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": 2,
                },
                {
                    "name": "predict_time",
                    "verbose_name": "Predict time (s)",
                    "value": None,
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": 2,
                },
            ],
            "ml_task": "regression",
            "related_items": [
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
                    "verbose_name": None,
                    "category": "model",
                    "attributes": {},
                    "parameters": {},
                    "representation": {"media_type": "text/html", "value": None},
                },
            ],
            "parameters": {"checksum": checksum},
            "key": "<key>",
            "run_id": "<run_id>",
        }

    def test_put_without_upload(self, respx_mock, regression):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": "<run_id>"})
        )
        respx_mock.post("projects/<tenant>/<name>/artefacts").mock(
            Response(200, json=[])
        )
        respx_mock.post("projects/<tenant>/<name>/items").mock(Response(200))

        Project("<tenant>", "<name>").put("<key>", regression)

        _, checksum = dumps(regression)

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
            "estimator_hyper_params": {},
            "metrics": [
                {
                    "name": "r2",
                    "verbose_name": "R²",
                    "value": None,
                    "data_source": "train",
                    "greater_is_better": True,
                    "position": None,
                },
                {
                    "name": "r2",
                    "verbose_name": "R²",
                    "value": None,
                    "data_source": "test",
                    "greater_is_better": True,
                    "position": None,
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
                    "name": "rmse",
                    "verbose_name": "RMSE",
                    "value": None,
                    "data_source": "test",
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
                    "data_source": "train",
                    "greater_is_better": False,
                    "position": 2,
                },
                {
                    "name": "predict_time",
                    "verbose_name": "Predict time (s)",
                    "value": None,
                    "data_source": "test",
                    "greater_is_better": False,
                    "position": 2,
                },
            ],
            "ml_task": "regression",
            "related_items": [
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
                    "verbose_name": None,
                    "category": "model",
                    "attributes": {},
                    "parameters": {},
                    "representation": {"media_type": "text/html", "value": None},
                },
            ],
            "parameters": {"checksum": checksum},
            "key": "<key>",
            "run_id": "<run_id>",
        }

    def test_reports(self, respx_mock):
        url = "projects/<tenant>/<name>/runs"
        respx_mock.post(url).mock(Response(200, json={"id": "<run_id>"}))

        project = Project("<tenant>", "<name>")

        assert isinstance(project.reports, SimpleNamespace)
        assert hasattr(project.reports, "get")
        assert hasattr(project.reports, "metadata")

    def test_reports_get(self, respx_mock, regression):
        # Mock hub routes that will be called
        url = "projects/<tenant>/<name>/runs"
        response = Response(200, json={"id": "<run_id>"})
        respx_mock.post(url).mock(response)

        url = "projects/<tenant>/<name>/experiments/estimator-reports/<report_id>"
        response = Response(200, json={"raw": {"checksum": "<checksum>"}})
        respx_mock.get(url).mock(response)

        url = "projects/<tenant>/<name>/artefacts/read"
        response = Response(200, json=[{"url": "http://url.com"}])
        respx_mock.get(url).mock(response)

        with io.BytesIO() as stream:
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
        respx_mock.post(url).mock(Response(200, json={"id": "<run_id_2>"}))

        url = "projects/<tenant>/<name>/experiments/estimator-reports"
        respx_mock.get(url).mock(
            Response(
                200,
                json=[
                    {
                        "id": "<report_id_0>",
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
                        "id": "<report_id_1>",
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
        metadata = project.reports.metadata()

        assert metadata == [
            {
                "id": "<report_id_0>",
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
                "id": "<report_id_1>",
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

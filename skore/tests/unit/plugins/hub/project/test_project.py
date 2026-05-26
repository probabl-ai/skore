from io import BytesIO
from json import dumps, loads

import joblib
from httpx import Client as HTTPXClient
from httpx import Response
from pytest import fixture, mark, raises, warns

from skore import CrossValidationReport, EstimatorReport
from skore._plugins.hub.artifact.pickle import Pickle
from skore._plugins.hub.artifact.upload import upload_artifacts
from skore._plugins.hub.exception import ForbiddenException, NotFoundException
from skore._plugins.hub.project.project import Project
from skore._plugins.hub.report import (
    CrossValidationReportPayload,
    EstimatorReportPayload,
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
def monkeypatch_table_report_representation(monkeypatch):
    monkeypatch.setattr(
        "skore._plugins.hub.artifact.media.data.TableReport.content_to_upload",
        lambda self: None,
    )


def _expected_payload_dump(Payload, project, key, report):
    """Build the expected wire JSON, mirroring what ``put`` does."""
    # Late import: ``monkeypatch_artifact_hub_client`` swaps the symbol in
    # the upload module's namespace.
    from skore._plugins.hub.artifact.upload import HubClient

    payload = Payload(project=project, key=key, report=report)
    media_artifacts = [
        media_cls(project=project, report=report) for media_cls in payload.MEDIAS
    ]
    pickle_artifact = Pickle(project=project, report=report)

    with HubClient() as hub_client, HTTPXClient() as storage_client:
        plans = upload_artifacts(
            hub_client=hub_client,
            storage_client=storage_client,
            workspace=project.workspace,
            project_name=project.name,
            artifacts=[*media_artifacts, pickle_artifact],
        )

    *media_plans, pickle_plan = plans
    data = payload.model_dump()
    data["medias"] = [
        {
            "content_type": media.content_type,
            "name": media.name,
            "data_source": media.data_source,
            "checksum": plan.checksum,
        }
        for media, plan in zip(media_artifacts, media_plans, strict=True)
        if plan is not None
    ]
    data["pickle"] = {
        "content_type": pickle_artifact.content_type,
        "checksum": pickle_plan.checksum if pickle_plan is not None else None,
    }
    return loads(dumps(data))


@mark.filterwarnings(
    "ignore:.*The workspace name can only contain unicode.*:UserWarning"
)
@mark.filterwarnings("ignore:.*The project name can only contain unicode.*:UserWarning")
@mark.usefixtures("monkeypatch_project_hub_client")
@mark.usefixtures("monkeypatch_artifact_hub_client")
class TestProject:
    @mark.respx()
    def test_workspace(self, respx_mock):
        mocks = [
            ("get", "/projects/available", Response(200)),
            (
                "post",
                "/projects/available/name",
                Response(200, json={"id": 42, "url": "http://domain/workspace/name"}),
            ),
            ("get", "/projects/unavailable", Response(404)),
            ("get", "/projects/forbidden", Response(403)),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        assert Project(workspace="available", name="name").workspace == "available"

        with raises(NotFoundException, match="not found"):
            Project(workspace="unavailable", name="name")

        with raises(ForbiddenException, match="not member"):
            Project(workspace="forbidden", name="name")

    @mark.parametrize(
        "input,output,warning",
        (
            ("myname", "myname", False),
            ("my.name", "my.name", False),
            ("my-name", "my-name", False),
            ("my_name", "my_name", False),
            ("my name", "my-name", True),
            ("mynäme", "myname", True),
            ("my/name", "my-name", True),
            ("my:name", "my-name", True),
            ("my?name", "my-name", True),
            ("my#name", "my-name", True),
            ("my/:?#näme", "my-name", True),
            ("👽👾😸👨 ? # @ mÿ-name ßß œπ æµ∂ƒ", "my-name", True),
        ),
    )
    @mark.respx(assert_all_called=False)
    def test_name(self, input, output, warning, respx_mock):
        post_response = Response(
            201,
            json={"id": 42, "url": "http://domain/myworkspace/myname"},
        )
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            ("post", "/projects/workspace/myname", post_response),
            ("post", "/projects/workspace/my-name", post_response),
            ("post", "/projects/workspace/my.name", post_response),
            ("post", "/projects/workspace/my_name", post_response),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        if warning:
            with warns(UserWarning, match=f".*'{output}'.*"):
                assert Project(workspace="workspace", name=input).name == output
        else:
            assert Project(workspace="workspace", name=input).name == output

    @mark.respx()
    def test_name_empty(self, respx_mock):
        mocks = [("get", "/projects/workspace", Response(200))]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        with raises(ValueError, match="Project name must not be empty."):
            Project(workspace="workspace", name="")

        with (
            raises(ValueError, match="Project name must not be empty."),
            warns(UserWarning, match="Your project will be created as ''"),
        ):
            Project(workspace="workspace", name="あいうえお")

    @mark.respx()
    def test_name_too_long(self, respx_mock):
        mocks = [("get", "/projects/workspace", Response(200))]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        with raises(
            ValueError, match="Project name must be no more than 64 characters long."
        ):
            Project(workspace="workspace", name=("a" * 500))

    @mark.respx()
    def test_put_exception(
        self,
        respx_mock,
    ):
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        with raises(TypeError, match="Key must be a string"):
            Project(workspace="workspace", name="name").put(None, "<value>")

        with raises(
            TypeError,
            match="must be a `skore.EstimatorReport` or `skore.CrossValidationReport`",
        ):
            Project(workspace="workspace", name="name").put("<key>", "<value>")

    @mark.respx()
    def test_put_estimator_report(self, monkeypatch, binary_classification, respx_mock):
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
            ("post", "projects/workspace/name/artifacts", Response(200, json=[])),
            (
                "post",
                "projects/workspace/name/estimator-reports",
                Response(201, json={"id": 42}),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        project.put("<key>", binary_classification)

        content = loads(respx_mock.calls.last.request.content.decode())
        desired = _expected_payload_dump(
            EstimatorReportPayload, project, "<key>", binary_classification
        )
        assert content == desired

    @mark.filterwarnings(
        # ignore precision warning due to the low number of labels in
        # `small_cv_binary_classification`, raised by `scikit-learn`
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning",
        # ignore deprecation warnings generated by the way `pandas` is used by
        # `searborn` and `skore`
        "ignore:The default of observed=False is deprecated.*:FutureWarning",
    )
    @mark.respx()
    def test_put_cross_validation_report(
        self, monkeypatch, small_cv_binary_classification, respx_mock
    ):
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
            ("post", "projects/workspace/name/artifacts", Response(200, json=[])),
            (
                "post",
                "projects/workspace/name/cross-validation-reports",
                Response(200, json={"id": 42}),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        project.put("<key>", small_cv_binary_classification)

        # Retrieve the content of the request
        content = loads(respx_mock.calls.last.request.content.decode())
        desired = _expected_payload_dump(
            CrossValidationReportPayload,
            project,
            "<key>",
            small_cv_binary_classification,
        )
        assert content == desired

    @mark.respx()
    def test_put_estimator_report_string_labels_with_pos_label(
        self, binary_classification_string_labels_with_pos_label, respx_mock
    ):
        """Put with binary string labels and pos_label set works."""
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
            ("post", "projects/workspace/name/artifacts", Response(200, json=[])),
            (
                "post",
                "projects/workspace/name/estimator-reports",
                Response(200, json={"id": 42}),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        report = binary_classification_string_labels_with_pos_label
        project.put("<key>", report)

        content = loads(respx_mock.calls.last.request.content.decode())
        desired = _expected_payload_dump(
            EstimatorReportPayload, project, "<key>", report
        )
        assert content == desired

    @mark.respx()
    def test_put_estimator_report_string_labels_pos_label_none(
        self, binary_classification_string_labels, respx_mock
    ):
        """Put with binary string labels and default ``pos_label`` (None) works."""
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
            ("post", "projects/workspace/name/artifacts", Response(200, json=[])),
            (
                "post",
                "projects/workspace/name/estimator-reports",
                Response(200, json={"id": 42}),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        report = binary_classification_string_labels
        project.put("<key>", report)

        content = loads(respx_mock.calls.last.request.content.decode())
        desired = _expected_payload_dump(
            EstimatorReportPayload, project, "<key>", report
        )
        assert content == desired

    @mark.filterwarnings(
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning",
        "ignore:The default of observed=False is deprecated.*:FutureWarning",
    )
    @mark.respx()
    def test_put_cross_validation_report_string_labels_pos_label_none(
        self, cv_binary_classification_string_labels, respx_mock
    ):
        """Put with CV binary string labels and default ``pos_label`` (None) works."""
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
            ("post", "projects/workspace/name/artifacts", Response(200, json=[])),
            (
                "post",
                "projects/workspace/name/cross-validation-reports",
                Response(201, json={"id": 42}),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        report = cv_binary_classification_string_labels
        project.put("<key>", report)

        content = loads(respx_mock.calls.last.request.content.decode())
        desired = _expected_payload_dump(
            CrossValidationReportPayload, project, "<key>", report
        )
        assert content == desired

    @mark.filterwarnings(
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning",
        "ignore:The default of observed=False is deprecated.*:FutureWarning",
    )
    @mark.respx()
    def test_put_cross_validation_report_string_labels_with_pos_label(
        self, cv_binary_classification_string_labels_with_pos_label, respx_mock
    ):
        """Put with CV binary string labels and pos_label set works."""
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
            ("post", "projects/workspace/name/artifacts", Response(200, json=[])),
            (
                "post",
                "projects/workspace/name/cross-validation-reports",
                Response(201, json={"id": 42}),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        report = cv_binary_classification_string_labels_with_pos_label
        project.put("<key>", report)

        content = loads(respx_mock.calls.last.request.content.decode())
        desired = _expected_payload_dump(
            CrossValidationReportPayload, project, "<key>", report
        )
        assert content == desired

    @mark.respx()
    def test_get_estimator_report(self, respx_mock, regression):
        with BytesIO() as stream:
            joblib.dump(regression, stream)

            mocks = [
                ("get", "/projects/workspace", Response(200)),
                (
                    "post",
                    "/projects/workspace/name",
                    Response(
                        201,
                        json={"id": 42, "url": "http://domain/myworkspace/myname"},
                    ),
                ),
                (
                    "get",
                    "projects/workspace/name/estimator-reports/<report_id>",
                    Response(200, json={"pickle": {"presigned_url": "http://url.com"}}),
                ),
                ("get", "http://url.com", Response(200, content=stream.getvalue())),
            ]

            for method, url, response in mocks:
                respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        report = project.get("skore:report:estimator:<report_id>")

        assert isinstance(report, EstimatorReport)
        assert report.estimator_name_ == regression.estimator_name_
        assert report.ml_task == regression.ml_task

    @mark.respx()
    def test_reports_get_cross_validation_report(self, respx_mock, cv_regression):
        with BytesIO() as stream:
            joblib.dump(cv_regression, stream)

            mocks = [
                ("get", "/projects/workspace", Response(200)),
                (
                    "post",
                    "/projects/workspace/name",
                    Response(
                        201,
                        json={"id": 42, "url": "http://domain/myworkspace/myname"},
                    ),
                ),
                (
                    "get",
                    "projects/workspace/name/cross-validation-reports/<report_id>",
                    Response(200, json={"pickle": {"presigned_url": "http://url.com"}}),
                ),
                ("get", "http://url.com", Response(200, content=stream.getvalue())),
            ]

            for method, url, response in mocks:
                respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        report = project.get("skore:report:cross-validation:<report_id>")

        assert isinstance(report, CrossValidationReport)
        assert report.estimator_name_ == cv_regression.estimator_name_
        assert report.ml_task == cv_regression.ml_task

    @mark.respx()
    def test_summarize(self, nowstr, respx_mock):
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
            (
                "get",
                "projects/workspace/name/estimator-reports/",
                Response(
                    200,
                    json=[
                        {
                            "urn": "skore:report:estimator:<report_id_0>",
                            "id": "<report_id_0>",
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
                            "urn": "skore:report:estimator:<report_id_1>",
                            "id": "<report_id_1>",
                            "key": "<key>",
                            "ml_task": "<ml_task>",
                            "estimator_class_name": "<estimator_class_name>",
                            "dataset_fingerprint": "<dataset_fingerprint>",
                            "created_at": nowstr,
                            "metrics": [
                                {
                                    "name": "log_loss",
                                    "value": 0,
                                    "data_source": "train",
                                },
                                {"name": "log_loss", "value": 2, "data_source": "test"},
                            ],
                        },
                    ],
                ),
            ),
            (
                "get",
                "projects/workspace/name/cross-validation-reports/",
                Response(
                    200,
                    json=[
                        {
                            "urn": "skore:report:cross-validation:<report_id_2>",
                            "id": "<report_id_2>",
                            "key": "<key>",
                            "ml_task": "<ml_task>",
                            "estimator_class_name": "<estimator_class_name>",
                            "dataset_fingerprint": "<dataset_fingerprint>",
                            "created_at": nowstr,
                            "metrics": [
                                {
                                    "name": "rmse_mean",
                                    "value": 0,
                                    "data_source": "train",
                                },
                                {
                                    "name": "rmse_mean",
                                    "value": 3,
                                    "data_source": "test",
                                },
                            ],
                        },
                    ],
                ),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        summary = project.summarize()

        assert summary == [
            {
                "id": "skore:report:estimator:<report_id_0>",
                "key": "<key>",
                "date": nowstr,
                "learner": "<estimator_class_name>",
                "ml_task": "<ml_task>",
                "report_type": "estimator",
                "dataset": "<dataset_fingerprint>",
                "rmse": 1,
                "log_loss": None,
                "roc_auc": None,
                "fit_time": None,
                "predict_time": None,
                "rmse_mean": None,
                "log_loss_mean": None,
                "roc_auc_mean": None,
                "fit_time_mean": None,
                "predict_time_mean": None,
            },
            {
                "id": "skore:report:estimator:<report_id_1>",
                "key": "<key>",
                "date": nowstr,
                "learner": "<estimator_class_name>",
                "ml_task": "<ml_task>",
                "report_type": "estimator",
                "dataset": "<dataset_fingerprint>",
                "rmse": None,
                "log_loss": 2,
                "roc_auc": None,
                "fit_time": None,
                "predict_time": None,
                "rmse_mean": None,
                "log_loss_mean": None,
                "roc_auc_mean": None,
                "fit_time_mean": None,
                "predict_time_mean": None,
            },
            {
                "id": "skore:report:cross-validation:<report_id_2>",
                "key": "<key>",
                "date": nowstr,
                "learner": "<estimator_class_name>",
                "ml_task": "<ml_task>",
                "report_type": "cross-validation",
                "dataset": "<dataset_fingerprint>",
                "rmse": None,
                "log_loss": None,
                "roc_auc": None,
                "fit_time": None,
                "predict_time": None,
                "rmse_mean": 3,
                "log_loss_mean": None,
                "roc_auc_mean": None,
                "fit_time_mean": None,
                "predict_time_mean": None,
            },
        ]

    @mark.respx
    def test_delete(self, respx_mock):
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            ("delete", "/projects/workspace/name", Response(204)),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        Project.delete(workspace="workspace", name="name")

    @mark.respx
    def test_delete_exception(self, respx_mock):
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            ("delete", "/projects/workspace/name", Response(403)),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        with raises(
            ForbiddenException,
            match=(
                "Failed to delete the project 'name'; "
                "please contact the 'workspace' owner"
            ),
        ):
            Project.delete(workspace="workspace", name="name")

    @mark.filterwarnings(
        # ignore deprecation warnings generated by the way `pandas` is used by
        # `searborn` and `skore`
        "ignore:The default of observed=False is deprecated.*:FutureWarning",
    )
    def test_put_reports_prints_console_message(
        self, monkeypatch, binary_classification, respx_mock, cv_binary_classification
    ):
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/myworkspace/myname"},
                ),
            ),
            ("post", "projects/workspace/name/artifacts", Response(200, json=[])),
            (
                "post",
                "projects/workspace/name/estimator-reports",
                Response(
                    201,
                    json={"id": 42},
                ),
            ),
            (
                "post",
                "projects/workspace/name/cross-validation-reports",
                Response(
                    201,
                    json={"id": 42},
                ),
            ),
        ]

        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")

        report_url = "http://domain/workspace/name/estimators/42"

        def assert_report_url(msg, *args, **kwargs):
            assert report_url in msg

        monkeypatch.setattr("rich.console.Console", assert_report_url)
        project.put("<key>", binary_classification)

        cv_report_url = "http://domain/workspace/name/cross-validations/42"

        def assert_cv_report_url(msg, *args, **kwargs):
            assert cv_report_url in msg

        monkeypatch.setattr("rich.console.Console", assert_cv_report_url)
        project.put("<key>", cv_binary_classification)

    @mark.respx()
    def test_put_makes_one_artifacts_call(self, respx_mock, binary_classification):
        """The whole point of Approach A: one batched POST /artifacts and one
        batched POST /artifacts/complete per ``put``, regardless of how many
        media artifacts the report produces."""
        mocks = [
            ("get", "/projects/workspace", Response(200)),
            (
                "post",
                "/projects/workspace/name",
                Response(
                    201,
                    json={"id": 42, "url": "http://domain/workspace/name"},
                ),
            ),
            ("post", "projects/workspace/name/artifacts", Response(201, json=[])),
            (
                "post",
                "projects/workspace/name/estimator-reports",
                Response(201, json={"id": 42}),
            ),
        ]
        for method, url, response in mocks:
            respx_mock.request(method=method, url=url).mock(response)

        project = Project(workspace="workspace", name="name")
        project.put("<key>", binary_classification)

        artifacts_calls = [
            c
            for c in respx_mock.calls
            if c.request.method == "POST" and c.request.url.path.endswith("/artifacts")
        ]
        complete_calls = [
            c
            for c in respx_mock.calls
            if c.request.url.path.endswith("/artifacts/complete")
        ]
        assert len(artifacts_calls) == 1, (
            f"expected exactly one batched POST /artifacts; got {len(artifacts_calls)}"
        )
        # With the mock returning [], no chunks are uploaded so /complete is
        # never called. The point is just that there's at most one.
        assert len(complete_calls) <= 1

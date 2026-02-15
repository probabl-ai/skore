from functools import partialmethod
from io import BytesIO
from json import dumps, loads
from urllib.parse import urljoin

import joblib
from httpx import Client, Response
from pytest import fixture, mark, raises, warns
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project.project.project import Project
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
        "skore_hub_project.artifact.upload.HUBClient",
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
        "skore.EstimatorReport.inspection.permutation_importance",
        partialmethod(
            skore.EstimatorReport.inspection.permutation_importance,
            seed=42,
        ),
    )


@fixture(autouse=True)
def monkeypatch_table_report_representation(monkeypatch):
    monkeypatch.setattr(
        "skore_hub_project.artifact.media.data.TableReport.content_to_upload",
        lambda self: None,
    )


@mark.filterwarnings(
    "ignore:.*The workspace name can only contain unicode.*:UserWarning"
)
@mark.filterwarnings("ignore:.*The project name can only contain unicode.*:UserWarning")
class TestProject:
    @mark.parametrize(
        "input,output,warning",
        (
            ("myworkspace", "myworkspace", False),
            ("my.workspace", "my.workspace", False),
            ("my-workspace", "my-workspace", False),
            ("my_workspace", "my_workspace", False),
            ("my workspace", "my-workspace", True),
            ("myworkspacë", "myworkspace", True),
            ("my/workspace", "my-workspace", True),
            ("my:workspace", "my-workspace", True),
            ("my?workspace", "my-workspace", True),
            ("my#workspace", "my-workspace", True),
            ("my/:?#workspacë", "my-workspace", True),
            ("👽👾😸👨 ? # @ mÿ-workspace ßß œπ æµ∂ƒ", "my-workspace", True),
        ),
    )
    def test_workspace(self, input, output, warning):
        if warning:
            with warns(UserWarning, match=f".*'{output}'.*"):
                assert Project(input, "myname").workspace == output
        else:
            assert Project(input, "myname").workspace == output

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
    def test_name(self, input, output, warning):
        if warning:
            with warns(UserWarning, match=f".*'{output}'.*"):
                assert Project("myworkspace", input).name == output
        else:
            assert Project("myworkspace", input).name == output

    def test_name_empty(self):
        err_msg = "Project name must not be empty."
        with raises(ValueError, match=err_msg):
            Project("myworkspace", "")

        warn_msg = "Your project will be created as ''"
        with warns(UserWarning, match=warn_msg), raises(ValueError, match=err_msg):
            Project("myworkspace", "あいうえお")

    def test_put_exception(
        self,
        respx_mock,
        binary_classification_string_labels,
        cv_binary_classification_string_labels,
    ):
        respx_mock.post("projects/myworkspace/myname").mock(Response(200))

        with raises(TypeError, match="Key must be a string"):
            Project("myworkspace", "myname").put(None, "<value>")

        with raises(
            TypeError,
            match="must be a `skore.EstimatorReport` or `skore.CrossValidationReport`",
        ):
            Project("myworkspace", "myname").put("<key>", "<value>")

        pos_label_msg = (
            "For binary classification, the positive label must be specified. "
            "You can set it using `report.pos_label = <positive_label>`."
        )
        with raises(ValueError, match=pos_label_msg):
            Project("myworkspace", "myname").put(
                "<key>", binary_classification_string_labels
            )
        with raises(ValueError, match=pos_label_msg):
            Project("myworkspace", "myname").put(
                "<key>", cv_binary_classification_string_labels
            )

    def test_put_estimator_report(self, monkeypatch, binary_classification, respx_mock):
        respx_mock.post("projects/myworkspace/myname").mock(Response(200))
        respx_mock.post("projects/myworkspace/myname/artifacts").mock(
            Response(200, json=[])
        )
        respx_mock.post("projects/myworkspace/myname/estimator-reports").mock(
            Response(200)
        )

        project = Project("myworkspace", "myname")
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

    @mark.filterwarnings(
        # ignore precision warning due to the low number of labels in
        # `small_cv_binary_classification`, raised by `scikit-learn`
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning",
        # ignore deprecation warnings generated by the way `pandas` is used by
        # `searborn`, which is a dependency of `skore`
        "ignore:The default of observed=False is deprecated.*:FutureWarning:seaborn",
    )
    def test_put_cross_validation_report(
        self, monkeypatch, small_cv_binary_classification, respx_mock
    ):
        respx_mock.post("projects/myworkspace/myname").mock(Response(200))
        respx_mock.post("projects/myworkspace/myname/artifacts").mock(
            Response(200, json=[])
        )
        respx_mock.post("projects/myworkspace/myname/cross-validation-reports").mock(
            Response(200)
        )

        project = Project("myworkspace", "myname")
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

    def test_put_estimator_report_string_labels_with_pos_label(
        self, binary_classification_string_labels_with_pos_label, respx_mock
    ):
        """Put with binary string labels and pos_label set works."""
        respx_mock.post("projects/myworkspace/myname").mock(Response(200))
        respx_mock.post("projects/myworkspace/myname/artifacts").mock(
            Response(200, json=[])
        )
        respx_mock.post("projects/myworkspace/myname/estimator-reports").mock(
            Response(200)
        )

        project = Project("myworkspace", "myname")
        report = binary_classification_string_labels_with_pos_label
        project.put("<key>", report)

        content = loads(respx_mock.calls.last.request.content.decode())
        desired = loads(
            dumps(
                EstimatorReportPayload(
                    project=project, key="<key>", report=report
                ).model_dump()
            )
        )
        assert content == desired

    @mark.filterwarnings(
        "ignore:Precision is ill-defined.*:sklearn.exceptions.UndefinedMetricWarning",
        "ignore:The default of observed=False is deprecated.*:FutureWarning:seaborn",
    )
    def test_put_cross_validation_report_string_labels_with_pos_label(
        self, cv_binary_classification_string_labels_with_pos_label, respx_mock
    ):
        """Put with CV binary string labels and pos_label set works."""
        respx_mock.post("projects/myworkspace/myname").mock(Response(200))
        respx_mock.post("projects/myworkspace/myname/artifacts").mock(
            Response(200, json=[])
        )
        respx_mock.post("projects/myworkspace/myname/cross-validation-reports").mock(
            Response(200)
        )

        project = Project("myworkspace", "myname")
        report = cv_binary_classification_string_labels_with_pos_label
        project.put("<key>", report)

        content = loads(respx_mock.calls.last.request.content.decode())
        desired = loads(
            dumps(
                CrossValidationReportPayload(
                    project=project, key="<key>", report=report
                ).model_dump()
            )
        )
        assert content == desired

    def test_get_estimator_report(self, respx_mock, regression):
        # Mock hub routes that will be called
        respx_mock.post("projects/myworkspace/myname").mock(Response(200))

        url = "projects/myworkspace/myname/estimator-reports/<report_id>"
        response = Response(200, json={"pickle": {"presigned_url": "http://url.com"}})
        respx_mock.get(url).mock(response)

        with BytesIO() as stream:
            joblib.dump(regression, stream)

            url = "http://url.com"
            response = Response(200, content=stream.getvalue())
            respx_mock.get(url).mock(response)

        # Test
        project = Project("myworkspace", "myname")
        report = project.get("skore:report:estimator:<report_id>")

        assert isinstance(report, EstimatorReport)
        assert report.estimator_name_ == regression.estimator_name_
        assert report.ml_task == regression.ml_task

    def test_reports_get_cross_validation_report(self, respx_mock, cv_regression):
        # Mock hub routes that will be called
        respx_mock.post("projects/myworkspace/myname").mock(Response(200))

        url = "projects/myworkspace/myname/cross-validation-reports/<report_id>"
        response = Response(200, json={"pickle": {"presigned_url": "http://url.com"}})
        respx_mock.get(url).mock(response)

        with BytesIO() as stream:
            joblib.dump(cv_regression, stream)

            url = "http://url.com"
            response = Response(200, content=stream.getvalue())
            respx_mock.get(url).mock(response)

        # Test
        project = Project("myworkspace", "myname")
        report = project.get("skore:report:cross-validation:<report_id>")

        assert isinstance(report, CrossValidationReport)
        assert report.estimator_name_ == cv_regression.estimator_name_
        assert report.ml_task == cv_regression.ml_task

    def test_summarize(self, nowstr, respx_mock):
        respx_mock.post("projects/myworkspace/myname").mock(Response(200))

        url = "projects/myworkspace/myname/estimator-reports/"
        respx_mock.get(url).mock(
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
                            {"name": "log_loss", "value": 0, "data_source": "train"},
                            {"name": "log_loss", "value": 2, "data_source": "test"},
                        ],
                    },
                ],
            )
        )

        url = "projects/myworkspace/myname/cross-validation-reports/"
        respx_mock.get(url).mock(
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
                            {"name": "rmse_mean", "value": 0, "data_source": "train"},
                            {"name": "rmse_mean", "value": 3, "data_source": "test"},
                        ],
                    },
                ],
            )
        )

        project = Project("myworkspace", "myname")
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

    def test_delete(self, respx_mock):
        respx_mock.delete("projects/myworkspace/myname").mock(Response(204))
        Project.delete("myworkspace", "myname")

    def test_delete_exception(self, respx_mock):
        respx_mock.delete("projects/myworkspace/myname").mock(Response(403))

        with raises(
            PermissionError,
            match=(
                "Failed to delete the project 'myname'; "
                "please contact the 'myworkspace' owner"
            ),
        ):
            Project.delete("myworkspace", "myname")

from datetime import timedelta
from urllib.parse import urljoin

from httpx import Client, Response
from pandas import DataFrame, MultiIndex, Index, RangeIndex
from pandas.testing import assert_index_equal
from pytest import fixture, mark
from skore_remote_project.project.project import Metadata
from skore_remote_project.item.skore_estimator_report_item import (
    SkoreEstimatorReportItem,
    Metadata as SkoreEstimatorReportItemMetadata,
)


class Namespace:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def request(self, method, url, **kwargs):
        response = super().request(method, urljoin("http://localhost", url), **kwargs)
        response.raise_for_status()

        return response


class TestMetadata:
    @fixture(autouse=True)
    def monkeypatch_client(self, monkeypatch):
        monkeypatch.setattr(
            "skore_remote_project.project.metadata.AuthenticatedClient",
            FakeClient,
        )

    @fixture(scope="class")
    def regression(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
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

    @fixture(scope="class")
    def binary_classification(self):
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from skore import EstimatorReport

        X, y = make_classification()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        return EstimatorReport(
            LogisticRegression(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

    @mark.respx(assert_all_called=True)
    def test_factory(self, now, respx_mock, regression, binary_classification):
        project = Namespace(tenant="<tenant>", name="<name>")
        url = "projects/<tenant>/<name>/experiments/estimator-reports"

        respx_mock.get(url).mock(
            Response(
                200,
                json=[
                    dict(
                        (
                            *SkoreEstimatorReportItemMetadata(regression),
                            ("id", 0),
                            ("run_id", 0),
                            ("created_at", now.isoformat()),
                        ),
                    ),
                    dict(
                        (
                            *SkoreEstimatorReportItemMetadata(binary_classification),
                            ("id", 1),
                            ("run_id", 1),
                            ("created_at", (now + timedelta(1)).isoformat()),
                        ),
                    ),
                ],
            )
        )

        metadata = Metadata.factory(project)

        assert isinstance(metadata, DataFrame)
        assert isinstance(metadata, Metadata)
        assert metadata.project == project
        assert_index_equal(
            metadata.index,
            MultiIndex.from_arrays(
                [
                    RangeIndex(2),
                    Index(["0", "1"], name="id", dtype=str),
                ]
            ),
        )
        assert list(metadata.columns) == [
            "run_id",
            "ml_task",
            "learner",
            "dataset",
            "date",
            "r2",
            "rmse",
            "fit_time",
            "predict_time_test",
            "accuracy",
            "brier_score",
            "log_loss",
            "roc_auc",
        ]

    @mark.respx(assert_all_called=True)
    def test_constructor(self, now, respx_mock, regression):
        project = Namespace(tenant="<tenant>", name="<name>")
        url = "projects/<tenant>/<name>/experiments/estimator-reports"

        respx_mock.get(url).mock(
            Response(
                200,
                json=[
                    dict(
                        (
                            *SkoreEstimatorReportItemMetadata(regression),
                            ("id", 0),
                            ("run_id", 0),
                            ("created_at", now.isoformat()),
                        ),
                    ),
                ],
            )
        )

        metadata = Metadata.factory(project)
        metadata2 = metadata.query("run_id==1")

        assert not DataFrame.equals(metadata2, metadata)
        assert isinstance(metadata2, DataFrame)
        assert isinstance(metadata2, Metadata)
        assert metadata2.project == project

    @mark.respx(assert_all_called=True)
    def test_reports_with_filter(
        self, monkeypatch, now, respx_mock, regression, binary_classification
    ):
        project = Namespace(tenant="<tenant>", name="<name>")
        item = SkoreEstimatorReportItem.factory(regression)
        url = "projects/<tenant>/<name>/experiments/estimator-reports"

        respx_mock.get(url).mock(
            Response(
                200,
                json=[
                    dict(
                        (
                            *SkoreEstimatorReportItemMetadata(regression),
                            ("id", 0),
                            ("run_id", 0),
                            ("created_at", now.isoformat()),
                        ),
                    ),
                    dict(
                        (
                            *SkoreEstimatorReportItemMetadata(binary_classification),
                            ("id", 1),
                            ("run_id", 1),
                            ("created_at", (now + timedelta(1)).isoformat()),
                        ),
                    ),
                ],
            )
        )

        url = "projects/<tenant>/<name>/experiments/estimator-reports/0"

        respx_mock.get(url).mock(
            Response(
                200,
                json={"raw": item.__parameters__["parameters"]},
            )
        )

        monkeypatch.setattr(
            "skore_remote_project.project.metadata.Metadata.query_string_selection",
            lambda self: "ml_task == 'regression'",
        )

        metadata = Metadata.factory(project)
        reports = metadata.reports()

        assert len(metadata) == 2
        assert isinstance(reports, list)
        assert len(reports) == 1
        assert type(reports[0]) is type(regression)
        assert reports[0].estimator_name_ == regression.estimator_name_
        assert reports[0].ml_task == regression.ml_task

    @mark.respx(assert_all_called=True)
    def test_reports_without_filter(self, now, respx_mock, regression):
        project = Namespace(tenant="<tenant>", name="<name>")
        item = SkoreEstimatorReportItem.factory(regression)
        url = "projects/<tenant>/<name>/experiments/estimator-reports"

        respx_mock.get(url).mock(
            Response(
                200,
                json=[
                    dict(
                        (
                            *SkoreEstimatorReportItemMetadata(regression),
                            ("id", 0),
                            ("run_id", 0),
                            ("created_at", now.isoformat()),
                        ),
                    ),
                ],
            )
        )

        url = "projects/<tenant>/<name>/experiments/estimator-reports/0"

        respx_mock.get(url).mock(
            Response(
                200,
                json={"raw": item.__parameters__["parameters"]},
            )
        )

        metadata = Metadata.factory(project)
        reports = metadata.reports(filter=False)

        assert len(metadata) == 1
        assert isinstance(reports, list)
        assert len(reports) == 1
        assert type(reports[0]) is type(regression)
        assert reports[0].estimator_name_ == regression.estimator_name_
        assert reports[0].ml_task == regression.ml_task

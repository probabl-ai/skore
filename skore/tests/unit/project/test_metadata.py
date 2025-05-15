from types import SimpleNamespace

from pandas import DataFrame, Index, MultiIndex, RangeIndex
from pandas.testing import assert_index_equal
from pytest import fixture
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore.project.metadata import Metadata


@fixture
def regression():
    X, y = make_regression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture
def binary_classification():
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


class FakeProject:
    def __init__(self, *reports):
        self.__reports = reports

    @property
    def reports(self):
        def get(id: str):
            return self.__reports[int(id)]

        def metadata():
            return [
                {
                    "id": index,
                    "run_id": None,
                    "key": None,
                    "date": None,
                    "learner": None,
                    "dataset": None,
                    "ml_task": report._ml_task,
                    "rmse": None,
                    "log_loss": None,
                    "roc_auc": None,
                    "fit_time": None,
                    "predict_time": None,
                }
                for index, report in enumerate(self.__reports)
            ]

        return SimpleNamespace(metadata=metadata, get=get)


class TestMetadata:
    def test_factory(self, regression, binary_classification):
        project = FakeProject(regression, binary_classification)
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
            "key",
            "date",
            "learner",
            "dataset",
            "ml_task",
            "rmse",
            "log_loss",
            "roc_auc",
            "fit_time",
            "predict_time",
        ]

    def test_factory_empty(self):
        project = FakeProject()
        metadata = Metadata.factory(project)

        assert isinstance(metadata, DataFrame)
        assert isinstance(metadata, Metadata)
        assert metadata.project == project

    def test_constructor(self, regression):
        project = FakeProject(regression)
        metadata = Metadata.factory(project)

        # Test with a bad query, with empty result
        metadata2 = metadata.query("ml_task=='<ml_task>'")

        assert isinstance(metadata2, DataFrame)
        assert isinstance(metadata2, Metadata)
        assert len(metadata2) == 0
        assert metadata2.project == project

        # Test with a valid query, with identical result
        metadata3 = metadata.query("ml_task=='regression'")

        assert isinstance(metadata3, DataFrame)
        assert isinstance(metadata3, Metadata)
        assert DataFrame.equals(metadata3, metadata)
        assert metadata3.project == project

    def test_reports_with_filter(self, monkeypatch, regression, binary_classification):
        project = FakeProject(regression, binary_classification)
        metadata = Metadata.factory(project)

        assert len(metadata) == 2
        assert metadata.reports() == [regression, binary_classification]

        monkeypatch.setattr(
            "skore.project.metadata.Metadata.query_string_selection",
            lambda self: "ml_task == 'regression'",
        )

        assert metadata.reports() == [regression]

    def test_reports_without_filter(
        self, monkeypatch, regression, binary_classification
    ):
        project = FakeProject(regression, binary_classification)
        metadata = Metadata.factory(project)

        assert len(metadata) == 2
        assert metadata.reports(filter=False) == [regression, binary_classification]

        monkeypatch.setattr(
            "skore.project.metadata.Metadata.query_string_selection",
            lambda self: "ml_task == 'regression'",
        )

        assert metadata.reports(filter=False) == [regression, binary_classification]

    def test_reports_empty(self):
        metadata = Metadata.factory(FakeProject())

        assert len(metadata) == 0
        assert metadata.reports() == []
        assert metadata.reports(filter=False) == []

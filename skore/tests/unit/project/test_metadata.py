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
        assert len(metadata) == 0

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
            "skore.project.metadata.Metadata._query_string_selection",
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
            "skore.project.metadata.Metadata._query_string_selection",
            lambda self: "ml_task == 'regression'",
        )

        assert metadata.reports(filter=False) == [regression, binary_classification]

    def test_reports_empty(self):
        metadata = Metadata.factory(FakeProject())

        assert len(metadata) == 0
        assert metadata.reports() == []
        assert metadata.reports(filter=False) == []

    def test__query_string_selection(self, monkeypatch):
        metadata = DataFrame(
            data={
                "ml_task": [
                    "classification",
                    "classification",
                    "classification",
                    "classification",
                    "regression",
                    "regression",
                    "regression",
                    "regression",
                ],
                "dataset": [
                    "dataset1",
                    "dataset1",
                    "dataset1",
                    "dataset2",
                    "dataset3",
                    "dataset3",
                    "dataset3",
                    "dataset4",
                ],
                "learner": [
                    "learner1",
                    "learner2",
                    "learner3",
                    "learner1",
                    "learner4",
                    "learner5",
                    "learner5",
                    "learner6",
                ],
                "fit_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "predict_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "rmse": [None, None, None, None, 0.1, 0.2, 0.3, 0.4],
                "log_loss": [0.3, 0.4, 0.5, 0.6, None, None, None, None],
                "roc_auc": [0.5, 0.6, 0.7, 0.8, None, None, None, None],
            },
            index=MultiIndex.from_tuples(
                [
                    (0, "id1"),
                    (0, "id2"),
                    (0, "id3"),
                    (0, "id4"),
                    (0, "id5"),
                    (0, "id6"),
                    (0, "id7"),
                    (0, "id8"),
                ],
                names=[None, "id"],
            ),
        )
        metadata["learner"] = metadata["learner"].astype("category")
        metadata = Metadata(metadata)
        metadata._repr_html_()  # trigger the creation of the widget

        expected_query = (
            "ml_task.str.contains('classification') and dataset == 'dataset1'"
        )
        assert metadata._query_string_selection() == expected_query

        # simulate a selection on the log loss dimension
        select_range_log_loss = {
            "ml_task": "classification",
            "dataset": "dataset1",
            "log_loss": (0.35, 0.55),
        }

        def mock_update_selection(*args, **kwargs):
            metadata._plot_widget.current_selection = select_range_log_loss
            return metadata._plot_widget

        monkeypatch.setattr(
            metadata._plot_widget, "update_selection", mock_update_selection
        )

        assert metadata._query_string_selection() == (
            "ml_task.str.contains('classification') and dataset == 'dataset1' "
            "and ((log_loss >= 0.350000 and log_loss <= 0.550000))"
        )

        # simulate a double selection on the log loss dimension
        select_range_log_loss = {
            "ml_task": "classification",
            "dataset": "dataset1",
            "log_loss": ((0.35, 0.45), (0.55, 0.55)),
        }

        def mock_update_selection(*args, **kwargs):
            metadata._plot_widget.current_selection = select_range_log_loss
            return metadata._plot_widget

        monkeypatch.setattr(
            metadata._plot_widget, "update_selection", mock_update_selection
        )

        assert metadata._query_string_selection() == (
            "ml_task.str.contains('classification') and dataset == 'dataset1' "
            "and ((log_loss >= 0.350000 and log_loss <= 0.450000) "
            "or (log_loss >= 0.550000 and log_loss <= 0.550000))"
        )

        # simulate a double selection on the log loss dimension and a selection of
        # learners
        select_range_log_loss = {
            "ml_task": "classification",
            "dataset": "dataset1",
            "log_loss": ((0.35, 0.45), (0.55, 0.55)),
            "learner": ((1e-18, 0.25), (1.5, 2.5)),
        }

        def mock_update_selection(*args, **kwargs):
            metadata._plot_widget.current_selection = select_range_log_loss
            return metadata._plot_widget

        monkeypatch.setattr(
            metadata._plot_widget, "update_selection", mock_update_selection
        )

        assert metadata._query_string_selection() == (
            "ml_task.str.contains('classification') and dataset == 'dataset1' "
            "and ((log_loss >= 0.350000 and log_loss <= 0.450000) "
            "or (log_loss >= 0.550000 and log_loss <= 0.550000)) "
            "and learner.isin(['learner1', 'learner3'])"
        )

from copy import deepcopy

from joblib import hash as joblib_hash
from pandas import DataFrame, Index, MultiIndex, RangeIndex
from pandas.testing import assert_index_equal
from pytest import fixture, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from skore._sklearn import ComparisonReport, CrossValidationReport, EstimatorReport
from skore.project._summary import Summary


@fixture
def estimator_report_regression():
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
def estimator_report_binary_classification():
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture
def cross_validation_report_regression():
    X, y = make_regression(random_state=42)
    return CrossValidationReport(LinearRegression(), X, y)


@fixture
def cross_validation_report_binary_classification():
    X, y = make_classification(random_state=42)
    return CrossValidationReport(LogisticRegression(), X, y)


class FakeProject:
    def __init__(self, *reports):
        self.__reports = reports

    def make_report_metadata(self, index, report):
        return {
            "id": index,
            "key": None,
            "date": None,
            "learner": None,
            "dataset": (
                joblib_hash(report.y_test)
                if isinstance(report, EstimatorReport)
                else joblib_hash(report.y)
            ),
            "ml_task": report._ml_task,
            "report_type": (
                "estimator"
                if isinstance(report, EstimatorReport)
                else "cross-validation"
            ),
            "rmse": None,
            "log_loss": None,
            "roc_auc": None,
            "fit_time": None,
            "predict_time": None,
            "rmse_mean": None,
            "log_loss_mean": None,
            "roc_auc_mean": None,
            "fit_time_mean": None,
            "predict_time_mean": None,
            "rmse_std": None,
            "log_loss_std": None,
            "roc_auc_std": None,
            "fit_time_std": None,
            "predict_time_std": None,
        }

    def get(self, id: str):
        return self.__reports[int(id)]

    def summarize(self):
        return [
            self.make_report_metadata(index, report)
            for index, report in enumerate(self.__reports)
        ]


class TestSummary:
    def test_factory(
        self,
        estimator_report_regression,
        estimator_report_binary_classification,
        cross_validation_report_regression,
        cross_validation_report_binary_classification,
    ):
        project = FakeProject(
            estimator_report_regression,
            estimator_report_binary_classification,
            cross_validation_report_regression,
            cross_validation_report_binary_classification,
        )
        summary = Summary.factory(project)

        assert isinstance(summary, DataFrame)
        assert isinstance(summary, Summary)
        assert summary.project == project
        assert_index_equal(
            summary.index,
            MultiIndex.from_arrays(
                [
                    RangeIndex(4),
                    Index(["0", "1", "2", "3"], name="id", dtype=str),
                ]
            ),
        )
        assert list(summary.columns) == [
            "key",
            "date",
            "learner",
            "dataset",
            "ml_task",
            "report_type",
            "rmse",
            "log_loss",
            "roc_auc",
            "fit_time",
            "predict_time",
            "rmse_mean",
            "log_loss_mean",
            "roc_auc_mean",
            "fit_time_mean",
            "predict_time_mean",
            "rmse_std",
            "log_loss_std",
            "roc_auc_std",
            "fit_time_std",
            "predict_time_std",
        ]

    def test_factory_empty(self):
        project = FakeProject()
        summary = Summary.factory(project)

        assert isinstance(summary, DataFrame)
        assert isinstance(summary, Summary)
        assert summary.project == project
        assert len(summary) == 0

    def test_constructor(self, estimator_report_regression):
        project = FakeProject(estimator_report_regression)
        summary = Summary.factory(project)

        # Test with a bad query, with empty result
        summary2 = summary.query("ml_task=='<ml_task>'")

        assert isinstance(summary2, DataFrame)
        assert isinstance(summary2, Summary)
        assert len(summary2) == 0
        assert summary2.project == project

        # Test with a valid query, with identical result
        summary3 = summary.query("ml_task=='regression'")

        assert isinstance(summary3, DataFrame)
        assert isinstance(summary3, Summary)
        assert DataFrame.equals(summary3, summary)
        assert summary3.project == project

    def test_reports_filter_true(
        self,
        monkeypatch,
        estimator_report_regression,
        estimator_report_binary_classification,
        cross_validation_report_regression,
        cross_validation_report_binary_classification,
    ):
        project = FakeProject(
            estimator_report_regression,
            estimator_report_binary_classification,
            cross_validation_report_regression,
            cross_validation_report_binary_classification,
        )
        summary = Summary.factory(project)

        assert summary.reports() == [
            estimator_report_regression,
            estimator_report_binary_classification,
            cross_validation_report_regression,
            cross_validation_report_binary_classification,
        ]

        monkeypatch.setattr(
            "skore.project._summary.Summary._query_string_selection",
            lambda self: "ml_task == 'regression'",
        )

        assert summary.reports() == [
            estimator_report_regression,
            cross_validation_report_regression,
        ]

    def test_reports_filter_false(
        self,
        monkeypatch,
        estimator_report_regression,
        estimator_report_binary_classification,
        cross_validation_report_regression,
        cross_validation_report_binary_classification,
    ):
        project = FakeProject(
            estimator_report_regression,
            estimator_report_binary_classification,
            cross_validation_report_regression,
            cross_validation_report_binary_classification,
        )
        summary = Summary.factory(project)

        assert summary.reports(filter=False) == [
            estimator_report_regression,
            estimator_report_binary_classification,
            cross_validation_report_regression,
            cross_validation_report_binary_classification,
        ]

        monkeypatch.setattr(
            "skore.project._summary.Summary._query_string_selection",
            lambda self: "ml_task == 'regression'",
        )

        assert summary.reports(filter=False) == [
            estimator_report_regression,
            estimator_report_binary_classification,
            cross_validation_report_regression,
            cross_validation_report_binary_classification,
        ]

    def test_reports_empty(self):
        summary = Summary.factory(FakeProject())

        assert len(summary) == 0
        assert summary.reports() == []
        assert summary.reports(filter=False) == []

    def test_reports_return_as_comparison(self, estimator_report_regression):
        regression1 = estimator_report_regression
        regression2 = deepcopy(estimator_report_regression)
        summary = Summary.factory(FakeProject(regression1, regression2))
        comparison = summary.reports(return_as="comparison")

        assert isinstance(comparison, ComparisonReport)
        assert comparison.reports_ == {
            "LinearRegression_1": regression1,
            "LinearRegression_2": regression2,
        }

    def test_reports_exception_invalid_object(self):
        with raises(
            RuntimeError,
            match="Bad condition: it is not a valid `Summary` object.",
        ):
            Summary([{"<column>": "<value>"}]).reports()

    def test_reports_exception_different_datasets(
        self, estimator_report_regression, estimator_report_binary_classification
    ):
        project = FakeProject(
            estimator_report_regression, estimator_report_binary_classification
        )
        summary = Summary.factory(project)

        with raises(
            RuntimeError,
            match=(
                "Bad condition: the comparison mode is only applicable when reports "
                "have the same dataset."
            ),
        ):
            summary.reports(return_as="comparison")

    def test__query_string_selection(self, monkeypatch):
        summary = DataFrame(
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
                "report_type": ["estimator"] * 8,
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
                "fit_time_mean": [None] * 8,
                "predict_time_mean": [None] * 8,
                "rmse_mean": [None] * 8,
                "log_loss_mean": [None] * 8,
                "roc_auc_mean": [None] * 8,
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
        summary["learner"] = summary["learner"].astype("category")
        summary = Summary(summary)
        summary._repr_mimebundle_()  # trigger the creation of the widget

        expected_query = (
            "ml_task.str.contains('classification') and dataset == 'dataset1'"
        )
        assert summary._query_string_selection() == expected_query

        # simulate a selection on the log loss dimension
        select_range_log_loss = {
            "ml_task": "classification",
            "dataset": "dataset1",
            "log_loss": (0.35, 0.55),
        }

        def mock_update_selection(*args, **kwargs):
            summary._plot_widget.current_selection = select_range_log_loss
            return summary._plot_widget

        monkeypatch.setattr(
            summary._plot_widget, "update_selection", mock_update_selection
        )

        assert summary._query_string_selection() == (
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
            summary._plot_widget.current_selection = select_range_log_loss
            return summary._plot_widget

        monkeypatch.setattr(
            summary._plot_widget, "update_selection", mock_update_selection
        )

        assert summary._query_string_selection() == (
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
            summary._plot_widget.current_selection = select_range_log_loss
            return summary._plot_widget

        monkeypatch.setattr(
            summary._plot_widget, "update_selection", mock_update_selection
        )

        assert summary._query_string_selection() == (
            "ml_task.str.contains('classification') and dataset == 'dataset1' "
            "and ((log_loss >= 0.350000 and log_loss <= 0.450000) "
            "or (log_loss >= 0.550000 and log_loss <= 0.550000)) "
            "and learner.isin(['learner1', 'learner3'])"
        )

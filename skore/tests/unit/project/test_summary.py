from copy import deepcopy

from joblib import hash as joblib_hash
from pandas import DataFrame, Index, MultiIndex, RangeIndex
from pandas.testing import assert_index_equal
from pytest import fixture, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from skore._project._summary import Summary
from skore._sklearn import ComparisonReport, CrossValidationReport, EstimatorReport


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
        is_estimator = isinstance(report, EstimatorReport)
        return {
            "id": index,
            "key": None,
            "date": None,
            "learner": report.estimator_name_,
            "dataset": (
                joblib_hash(report.y_test) if is_estimator else joblib_hash(report.y)
            ),
            "ml_task": report._ml_task,
            "report_type": "estimator" if is_estimator else "cross-validation",
            "rmse": 1.0 if (is_estimator and "regression" in report._ml_task) else None,
            "log_loss": (
                0.3 if (is_estimator and "classification" in report._ml_task) else None
            ),
            "roc_auc": (
                0.9 if (is_estimator and "classification" in report._ml_task) else None
            ),
            "fit_time": 0.1 if is_estimator else None,
            "predict_time": 0.01 if is_estimator else None,
            "rmse_mean": (
                1.0 if (not is_estimator and "regression" in report._ml_task) else None
            ),
            "log_loss_mean": (
                0.3
                if (not is_estimator and "classification" in report._ml_task)
                else None
            ),
            "roc_auc_mean": (
                0.9
                if (not is_estimator and "classification" in report._ml_task)
                else None
            ),
            "fit_time_mean": 0.5 if not is_estimator else None,
            "predict_time_mean": 0.05 if not is_estimator else None,
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
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = Summary.factory(project)

        assert isinstance(summary, Summary)
        assert summary.project is project
        assert_index_equal(
            summary.frame().index,
            MultiIndex.from_arrays(
                [
                    RangeIndex(2),
                    Index(["0", "1"], name="id", dtype=str),
                ]
            ),
        )

    def test_factory_empty(self):
        project = FakeProject()
        summary = Summary.factory(project)

        assert isinstance(summary, Summary)
        assert summary.project is project
        assert len(summary.frame()) == 0

    def test_frame_report_type(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = Summary.factory(project)

        # Without filtering, both rows are present.
        assert len(summary.frame()) == 2

        # ``ml_task`` is constant within a project, so it is dropped from ``frame``.
        assert "ml_task" not in summary.frame().columns

        # Estimator view keeps the raw metric columns and drops the empty ``_mean``.
        estimator_frame = summary.frame(report_type="estimator")
        assert len(estimator_frame) == 1
        assert "rmse" in estimator_frame.columns
        assert not any(col.endswith("_mean") for col in estimator_frame.columns)

        # Cross-validation view keeps the ``_mean`` columns and drops the raw metrics.
        cv_frame = summary.frame(report_type="cross-validation")
        assert len(cv_frame) == 1
        assert "rmse_mean" in cv_frame.columns
        assert "rmse" not in cv_frame.columns

    def test_query(self, estimator_report_regression):
        project = FakeProject(estimator_report_regression)
        summary = Summary.factory(project)

        # Query with no match.
        empty = summary.query("ml_task == '<ml_task>'")
        assert isinstance(empty, Summary)
        assert len(empty.frame()) == 0
        assert empty.project is project

        # Query matching everything.
        same = summary.query("ml_task == 'regression'")
        assert isinstance(same, Summary)
        assert DataFrame.equals(same.frame(), summary.frame())
        assert same.project is project

    def test_query_by_id(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = Summary.factory(project)

        selected = summary.query("id in ['0']")
        assert list(selected.frame().index.get_level_values("id")) == ["0"]
        assert selected.reports() == [estimator_report_regression]

    def test_reports(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = Summary.factory(project)

        assert summary.reports() == [
            estimator_report_regression,
            cross_validation_report_regression,
        ]

        assert summary.query("report_type == 'estimator'").reports() == [
            estimator_report_regression
        ]

    def test_reports_empty(self):
        summary = Summary.factory(FakeProject())

        assert len(summary.frame()) == 0
        assert summary.reports() == []

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
            Summary(DataFrame([{"<column>": "<value>"}])).reports()

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

    def test_html_repr(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = Summary.factory(project)

        html = summary._html_repr()

        # The single full-width table with per-row selection checkboxes.
        assert '<table class="summary-table">' in html
        assert 'class="skore-summary-row"' in html
        # Sortable headers with verbose labels (incl. the synthetic ``ID`` column).
        assert 'class="summary-sortable"' in html
        assert "data-sort-kind=" in html
        assert ">ID<" in html
        # The Filter button with a two-level menu: one entry per column, each
        # opening a scrollable submenu of values.
        assert 'class="skore-summary-filter-toggle"' in html
        assert 'class="skore-summary-filter-field-toggle"' in html
        assert 'class="skore-summary-filter-submenu"' in html
        assert 'class="skore-summary-filter-value"' in html
        assert 'data-field="report_type"' in html
        assert 'data-field="learner"' in html
        assert 'data-field="dataset"' in html
        assert 'data-filter-field="report_type"' in html
        assert 'data-filter-field="learner"' in html
        assert 'data-filter-field="dataset"' in html
        assert 'data-filter-value="estimator"' in html
        assert 'data-filter-value="cross-validation"' in html
        assert 'data-report-type="estimator"' in html
        assert 'data-report-type="cross-validation"' in html
        # The search bar and the Filter button share the toolbar row.
        assert 'class="skore-summary-toolbar"' in html
        assert 'class="skore-summary-search-input"' in html
        # The date-range filter (native start/end pickers).
        assert 'class="skore-summary-date-start"' in html
        assert 'class="skore-summary-date-end"' in html
        # The Group by menu: date (day/hour/custom), learner and estimator type.
        assert 'class="skore-summary-groupby-toggle"' in html
        assert 'data-group="date"' in html
        assert 'data-unit="day"' in html
        assert 'data-unit="hour"' in html
        assert 'data-unit="custom"' in html
        assert 'data-group="learner"' in html
        assert 'data-group="report_type"' in html
        assert 'data-group="none"' in html
        # Each row carries its full date for client-side filtering/grouping.
        assert "data-date=" in html
        # The query-string box, the inline SVG copy icon button and the help tooltip.
        assert 'class="skore-summary-query"' in html
        assert 'class="skore-summary-copy"' in html
        assert "<svg" in html
        assert 'class="report-tab-help skore-summary-help"' in html
        assert "tooltip-text" in html
        assert "skoreInitSummary" in html
        # ``ml_task`` is never displayed.
        assert "ml_task" not in html
        assert "Ml task" not in html

        mimebundle = summary._repr_mimebundle_()
        assert "text/html" in mimebundle
        assert "text/plain" in mimebundle

    def test_html_repr_empty(self):
        summary = Summary.factory(FakeProject())

        html = summary._html_repr()

        assert "No report found in the project" in html
        assert '<table class="summary-table">' not in html

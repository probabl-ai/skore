from copy import deepcopy

from joblib import hash as joblib_hash
from pandas import DataFrame, Index, MultiIndex, RangeIndex
from pandas.api.types import is_datetime64_any_dtype
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
            "rmse_std": (
                0.1 if (not is_estimator and "regression" in report._ml_task) else None
            ),
            "log_loss_std": (
                0.05
                if (not is_estimator and "classification" in report._ml_task)
                else None
            ),
            "roc_auc_std": (
                0.02
                if (not is_estimator and "classification" in report._ml_task)
                else None
            ),
            "fit_time_std": 0.02 if not is_estimator else None,
            "predict_time_std": 0.01 if not is_estimator else None,
        }

    def get(self, id: str):
        return self.__reports[int(id)]

    def summarize(self):
        return [
            self.make_report_metadata(index, report)
            for index, report in enumerate(self.__reports)
        ]


def _summary_from_project(project) -> Summary:
    """Mirror :meth:`Project.summarize` for unit tests using ``FakeProject``."""
    frame = DataFrame(project.summarize(), copy=False)
    if not frame.empty:
        frame.index = MultiIndex.from_arrays(
            [
                RangeIndex(len(frame)),
                Index(frame.pop("id"), name="id", dtype=str),
            ]
        )
    return Summary(frame, project)


class TestSummary:
    def test_summarize(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = _summary_from_project(project)

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

    def test_summarize_date_dtype(self, estimator_report_regression):
        project = FakeProject(estimator_report_regression)
        summary = _summary_from_project(project)

        assert is_datetime64_any_dtype(summary.frame()["date"])

    def test_summarize_empty(self):
        project = FakeProject()
        summary = _summary_from_project(project)

        assert isinstance(summary, Summary)
        assert summary.project is project
        assert len(summary.frame()) == 0

    def test_frame_drops_empty_object_metric_columns(self):
        frame = DataFrame(
            {
                "key": ["k0"],
                "date": ["2024-01-01"],
                "learner": ["LogisticRegression"],
                "dataset": ["d0"],
                "ml_task": ["binary-classification"],
                "report_type": ["cross-validation"],
                "rmse": [None],
                "rmse_mean": [None],
                "rmse_std": [None],
                "log_loss_mean": [0.3],
            }
        )
        frame.index = MultiIndex.from_arrays(
            [RangeIndex(1), Index(["cv0"], name="id", dtype=str)]
        )
        out = Summary(frame).frame()

        assert "rmse" not in out.columns
        assert "rmse_std" not in out.columns
        assert "log_loss" in out.columns

    def test_frame_keeps_std_when_populated(self):
        frame = DataFrame(
            {
                "key": ["k0"],
                "date": ["2024-01-01"],
                "learner": ["LogisticRegression"],
                "dataset": ["d0"],
                "ml_task": ["regression"],
                "report_type": ["cross-validation"],
                "rmse_mean": [1.0],
                "rmse_std": [0.1],
            }
        )
        frame.index = MultiIndex.from_arrays(
            [RangeIndex(1), Index(["cv0"], name="id", dtype=str)]
        )
        out = Summary(frame).frame()

        assert "rmse" in out.columns
        assert "rmse_std" in out.columns

    def test_html_repr_marks_std_columns_hidden(self):
        frame = DataFrame(
            {
                "key": ["k0"],
                "date": ["2024-01-01"],
                "learner": ["LogisticRegression"],
                "dataset": ["d0"],
                "ml_task": ["regression"],
                "report_type": ["cross-validation"],
                "rmse_mean": [1.0],
                "rmse_std": [0.1],
            }
        )
        frame.index = MultiIndex.from_arrays(
            [RangeIndex(1), Index(["cv0"], name="id", dtype=str)]
        )
        html = Summary(frame)._html_repr()

        assert 'data-column-role="std"' in html
        assert 'data-column-key="rmse_std"' in html
        rmse_std_toggle = html.split('data-column-key="rmse_std"')[1].split(">")[0]
        assert 'data-column-role="std"' in rmse_std_toggle
        assert "checked" not in rmse_std_toggle

    def test_frame_report_type(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = _summary_from_project(project)

        # Without filtering, both rows are present.
        assert len(summary.frame()) == 2

        # ``ml_task`` is constant within a project, so it is dropped from ``frame``.
        assert "ml_task" not in summary.frame().columns

        # Estimator view keeps the raw metric columns and drops the empty ``_mean``.
        estimator_frame = summary.frame(report_type="estimator")
        assert len(estimator_frame) == 1
        assert "rmse" in estimator_frame.columns
        assert not any(col.endswith("_mean") for col in estimator_frame.columns)

        # Cross-validation view exposes the merged metric column.
        cv_frame = summary.frame(report_type="cross-validation")
        assert len(cv_frame) == 1
        assert "rmse" in cv_frame.columns
        assert "rmse_mean" not in cv_frame.columns

    def test_summarize_merges_mean_metrics(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        frame = _summary_from_project(project).frame()

        assert "rmse" in frame.columns
        assert "rmse_mean" not in frame.columns
        assert frame["rmse"].notna().all()

    def test_html_repr_emits_data_std(self):
        frame = DataFrame(
            {
                "key": [None],
                "date": ["2024-01-01"],
                "learner": ["LinearRegression"],
                "dataset": ["d1"],
                "ml_task": ["regression"],
                "report_type": ["cross-validation"],
                "rmse": [1.0],
                "rmse_std": [0.1],
            }
        )
        frame.index = MultiIndex.from_arrays(
            [RangeIndex(1), Index(["cv0"], name="id", dtype=str)]
        )
        html = Summary(frame)._html_repr()

        assert 'data-std="0.1"' in html

    def test_html_repr_no_data_std_for_estimator(self, estimator_report_regression):
        html = _summary_from_project(
            FakeProject(estimator_report_regression)
        )._html_repr()

        assert 'data-std="' not in html

    def test_html_repr_sends_raw_numbers(self):
        frame = DataFrame(
            {
                "key": ["k0", "k1"],
                "date": ["2024-01-01", "2024-01-02"],
                "learner": ["LinearRegression", "LinearRegression"],
                "dataset": ["d0", "d0"],
                "ml_task": ["regression", "regression"],
                "report_type": ["cross-validation", "cross-validation"],
                "rmse_mean": [1.0, 2.0],
                "rmse_std": [0.1, None],
            }
        )
        frame.index = MultiIndex.from_arrays(
            [RangeIndex(2), Index(["cv0", "cv1"], name="id", dtype=str)]
        )
        html = Summary(frame)._html_repr()

        # Numbers are emitted raw in ``data-sort`` with an empty cell body; the
        # client renders the visible value and the NA fallback.
        assert 'data-sort="1.0" data-std="0.1"></td>' in html
        # ``data-std`` is always present for a metric paired with ``*_std`` and
        # carries the raw value (``nan`` for NA), leaving the rendering to the
        # client.
        assert 'data-sort="2.0" data-std="nan"></td>' in html

    def test_query(self, estimator_report_regression):
        project = FakeProject(estimator_report_regression)
        summary = _summary_from_project(project)

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
        summary = _summary_from_project(project)

        selected = summary.query("id in ['0']")
        assert list(selected.frame().index.get_level_values("id")) == ["0"]
        assert selected.compare() == [estimator_report_regression]

    def test_compare(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = _summary_from_project(project)

        assert summary.compare() == [
            estimator_report_regression,
            cross_validation_report_regression,
        ]

        assert summary.query("report_type == 'estimator'").compare() == [
            estimator_report_regression
        ]

    def test_compare_empty(self):
        summary = _summary_from_project(FakeProject())

        assert len(summary.frame()) == 0
        assert summary.compare() == []

    def test_compare_return_as_report(self, estimator_report_regression):
        regression1 = estimator_report_regression
        regression2 = deepcopy(estimator_report_regression)
        summary = _summary_from_project(FakeProject(regression1, regression2))
        comparison = summary.compare(return_as="report")

        assert isinstance(comparison, ComparisonReport)
        assert comparison.reports_ == {
            "LinearRegression_1": regression1,
            "LinearRegression_2": regression2,
        }

    def test_compare_exception_different_datasets(
        self, estimator_report_regression, estimator_report_binary_classification
    ):
        project = FakeProject(
            estimator_report_regression, estimator_report_binary_classification
        )
        summary = _summary_from_project(project)

        with raises(
            RuntimeError,
            match=(
                "Bad condition: the report mode is only applicable when reports "
                "have the same dataset."
            ),
        ):
            summary.compare(return_as="report")

    def test_plot_not_implemented(self):
        with raises(NotImplementedError):
            Summary(DataFrame()).plot()

    def test_html_repr(
        self,
        estimator_report_regression,
        cross_validation_report_regression,
    ):
        project = FakeProject(
            estimator_report_regression,
            cross_validation_report_regression,
        )
        summary = _summary_from_project(project)

        html = summary._html_repr()

        # The single full-width table with per-row selection checkboxes.
        assert '<table class="summary-table">' in html
        assert 'class="skore-summary-row"' in html
        # Sortable headers with verbose labels (incl. the synthetic ``ID`` column).
        assert 'class="summary-sortable"' in html
        assert "data-sort-kind=" in html
        # The column key is exposed so the plot can build a pandas query.
        assert "data-column-key=" in html
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
        # Date and ellipsis formatting is applied on the client (see summary.js).
        assert "SKORE_SUMMARY_ELLIPSIS_COLUMNS" in html
        assert 'class="skore-summary-filter-option-label"' in html
        assert "function middleEllipsis" in html
        assert "function formatDate" in html
        # Dataset filter options expose the full hash in the label (not pre-truncated).
        dataset_hash = html.split('data-dataset="')[1].split('"')[0]
        assert (
            f'<span class="skore-summary-filter-option-label">{dataset_hash}</span>'
            in html
        )
        # The query-string box, the inline SVG copy icon button and the help tooltip.
        assert 'class="skore-summary-query"' in html
        assert 'class="skore-summary-copy"' in html
        # The clear-selection button (icon + tooltip).
        assert 'class="skore-summary-clear"' in html
        assert 'aria-label="Clear selection"' in html
        assert "<svg" in html
        assert 'class="report-tab-help skore-summary-help"' in html
        assert "tooltip-text" in html
        assert "skoreInitSummary" in html
        # The Table/Plot view toggle and the parallel-coordinates plot container
        # (matched on the markup rather than the bundled CSS class definitions).
        assert '<div class="summary-view-toggle"' in html
        assert 'data-view="plot" aria-pressed="false" aria-label="Plot view"' in html
        assert '<div class="summary-plot-wrap">' in html
        assert '<div class="summary-plot">' in html
        assert '<select class="skore-summary-color-metric"' in html
        # The third view toggle and the trend-plot container (metric over time),
        # with its Metric selector in the controls row and per-row key for tooltips.
        assert 'data-view="trend" aria-pressed="false" aria-label="Trend view"' in html
        assert '<div class="summary-trend-wrap">' in html
        assert '<div class="summary-trend">' in html
        assert 'class="skore-summary-trend-metric-field"' in html
        assert 'aria-label="Metric"' in html
        assert html.index('class="skore-summary-trend-metric-field"') < html.index(
            '<div class="summary-table-wrap">'
        )
        assert '<select class="skore-summary-trend-metric"' in html
        assert 'class="summary-trend-undated-empty"' in html
        assert "No dated reports to plot." in html
        assert "data-key=" in html
        # Inline SVG icons replace the text labels / Font Awesome (no external dep).
        assert 'class="summary-select-col"' in html
        assert 'class="summary-sort-icon summary-sort-icon--asc"' in html
        # The columns menu lets the user toggle which columns are shown.
        assert 'class="skore-summary-columns-toggle"' in html
        assert 'aria-label="Show columns"' in html
        assert 'class="skore-summary-column-toggle"' in html
        # Default-hidden columns are configured in Python (hidden_by_default).
        panel_start = html.index('class="skore-summary-columns-panel"')
        panel_end = html.index("</div>", panel_start)
        panel = html[panel_start:panel_end]
        inputs = [
            chunk
            for chunk in panel.split("<input")
            if "skore-summary-column-toggle" in chunk
        ]
        assert inputs, "expected at least one column toggle"

        def toggle_attrs(key: str) -> str:
            return html.split(f'data-column-key="{key}"')[1].split(">")[0]

        for key in ("learner", "dataset", "report_type"):
            assert "checked" not in toggle_attrs(key)
        for key in ("rmse_std", "fit_time_std", "predict_time_std"):
            assert "checked" not in toggle_attrs(key)
        for key in ("id", "key", "date", "rmse", "fit_time", "predict_time"):
            assert "checked" in toggle_attrs(key)
        # ``date`` is rendered after metrics; default-hidden columns follow ``date``.
        assert html.index('data-column-key="date"') > html.index(
            'data-column-key="fit_time"'
        )
        assert (
            html.index('data-column-key="fit_time"')
            < html.index('data-column-key="date"')
            < html.index('data-column-key="learner"')
        )
        # ``ml_task`` is never displayed.
        assert "ml_task" not in html
        assert "Ml task" not in html

        mimebundle = summary._repr_mimebundle_()
        assert "text/html" in mimebundle
        assert "text/plain" in mimebundle

    def test_html_repr_empty(self):
        summary = _summary_from_project(FakeProject())

        html = summary._html_repr()

        assert "No report found in the project" in html
        assert '<table class="summary-table">' not in html

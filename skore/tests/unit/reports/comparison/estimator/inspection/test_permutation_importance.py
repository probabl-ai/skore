import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import ComparisonReport, EstimatorReport, PermutationImportanceDisplay
from skore._utils._testing import check_cache_changed


def _children_cache_size(report):
    sizes = {
        len(estimator_report._cache) for estimator_report in report.reports_.values()
    }
    msg = "In this test, we expect all children report to have the same cache size"
    assert len(sizes) == 1, msg
    (size,) = sizes
    return size


@pytest.fixture
def comparison_report_linear(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    return ComparisonReport(reports={"report_1": report_1, "report_2": report_2})


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        make_pipeline(StandardScaler(), LinearRegression()),
    ],
)
def test_returns_display(regression_train_test_split, estimator):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.inspection, "permutation_importance")
    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    assert isinstance(display, PermutationImportanceDisplay)


def test_cache_behavior(comparison_report_linear):
    report = comparison_report_linear
    assert _children_cache_size(report) == 0

    child_report = next(iter(report.reports_.values()))
    with check_cache_changed(child_report._cache):
        report.inspection.permutation_importance(seed=42, n_repeats=2)

    assert report._cache == {}
    assert _children_cache_size(report) == 1


def test_cache_seed_int(comparison_report_linear):
    report = comparison_report_linear
    assert report._cache == {}
    assert _children_cache_size(report) == 0

    display_1 = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test"
    )
    assert report._cache == {}
    assert _children_cache_size(report) == 1

    display_2 = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test"
    )
    assert display_1.importances.equals(display_2.importances)
    assert report._cache == {}
    assert _children_cache_size(report) == 1


def test_cache_seed_none(comparison_report_linear):
    report = comparison_report_linear
    assert _children_cache_size(report) == 0

    report.inspection.permutation_importance(n_repeats=2, data_source="test")
    assert _children_cache_size(report) == 1

    report.inspection.permutation_importance(n_repeats=2, data_source="test")
    assert _children_cache_size(report) == 1


def test_cache_parameter_in_cache(comparison_report_linear):
    report = comparison_report_linear
    report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test", metric="r2"
    )

    child_report = next(iter(report.reports_.values()))
    with check_cache_changed(child_report._cache):
        report.inspection.permutation_importance(
            seed=42,
            n_repeats=2,
            data_source="test",
            metric=make_scorer(root_mean_squared_error),
        )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(comparison_report_linear, data_source):
    report = comparison_report_linear
    display = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source=data_source
    )
    assert set(display.importances["data_source"]) == {data_source}


def test_seed_wrong_type(comparison_report_linear):
    report = comparison_report_linear
    with pytest.raises(
        ValueError, match="seed must be an integer or None; got <class 'str'>"
    ):
        report.inspection.permutation_importance(seed="42")

import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import ComparisonReport, EstimatorReport, PermutationImportanceDisplay
from skore._utils._testing import check_cache_changed


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
    assert report._cache == {}

    with check_cache_changed(report._cache):
        report.inspection.permutation_importance(seed=42, n_repeats=2)

    assert len(report._cache) == 1


def test_cache_seed_int(comparison_report_linear):
    report = comparison_report_linear
    assert report._cache == {}

    display_1 = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test"
    )
    assert len(report._cache) == 1

    display_2 = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test"
    )
    assert display_1.importances.equals(display_2.importances)
    assert len(report._cache) == 1


def test_cache_seed_none(comparison_report_linear):
    report = comparison_report_linear
    assert report._cache == {}

    report.inspection.permutation_importance(n_repeats=2, data_source="test")
    assert len(report._cache) == 1

    display_2 = report.inspection.permutation_importance(
        n_repeats=2, data_source="test"
    )
    assert len(report._cache) == 1
    cached_display = next(iter(report._cache.values()))
    assert cached_display is display_2


def test_cache_parameter_in_cache(comparison_report_linear):
    report = comparison_report_linear
    report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test", metric="r2"
    )

    with check_cache_changed(report._cache):
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


def test_data_source_X_y(comparison_report_linear, regression_data):
    X, y = regression_data
    report = comparison_report_linear
    display = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="X_y", X=X, y=y
    )
    assert isinstance(display, PermutationImportanceDisplay)
    assert set(display.importances["data_source"]) == {"X_y"}


def test_seed_wrong_type(comparison_report_linear):
    report = comparison_report_linear
    with pytest.raises(
        ValueError, match="seed must be an integer or None; got <class 'str'>"
    ):
        report.inspection.permutation_importance(seed="42")

import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import ComparisonReport, CrossValidationReport, PermutationImportanceDisplay
from skore._utils._testing import check_cache_changed


def _children_cache_size(report):
    sizes = {
        len(estimator_report._cache)
        for cv_report in report.reports_.values()
        for estimator_report in cv_report.estimator_reports_
    }
    msg = "In this test, we expect all children report to have the same cache size"
    assert len(sizes) == 1, msg
    (size,) = sizes
    return size


@pytest.fixture
def comparison_cv_report_ridge(regression_data):
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    return ComparisonReport(reports={"report_1": report_1, "report_2": report_2})


@pytest.mark.parametrize(
    "data_fixture, estimator",
    [
        ("regression_data", Ridge()),
        ("regression_data", make_pipeline(StandardScaler(), Ridge())),
        ("binary_classification_data", LogisticRegression()),
        (
            "binary_classification_data",
            make_pipeline(StandardScaler(), LogisticRegression()),
        ),
    ],
)
def test_returns_display(data_fixture, estimator, request):
    X, y = request.getfixturevalue(data_fixture)
    report_1 = CrossValidationReport(estimator, X, y, splitter=2)
    report_2 = CrossValidationReport(estimator, X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.inspection, "permutation_importance")
    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    assert isinstance(display, PermutationImportanceDisplay)


def test_split_column(comparison_cv_report_ridge):
    report = comparison_cv_report_ridge
    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    assert set(display.importances["split"]) == {0, 1}


def test_cache_behavior(comparison_cv_report_ridge):
    report = comparison_cv_report_ridge
    assert _children_cache_size(report) == 0

    child_report = next(iter(report.reports_.values())).estimator_reports_[0]
    with check_cache_changed(child_report._cache):
        report.inspection.permutation_importance(seed=42, n_repeats=2)

    assert _children_cache_size(report) == 1


def test_at_step(regression_data):
    X, y = regression_data
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), Ridge())
    report_1 = CrossValidationReport(pipeline, X, y, splitter=2)
    report_2 = CrossValidationReport(pipeline, X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display_raw = report.inspection.permutation_importance(
        seed=42, n_repeats=2, at_step=0
    )
    display_pca = report.inspection.permutation_importance(
        seed=42, n_repeats=2, at_step=-1
    )
    raw_features = set(display_raw.importances["feature"])
    pca_features = set(display_pca.importances["feature"])
    assert raw_features != pca_features
    assert len(raw_features) == 4
    assert len(pca_features) == 2


def test_cache_seed_int(comparison_cv_report_ridge):
    report = comparison_cv_report_ridge
    assert _children_cache_size(report) == 0

    display_1 = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test"
    )
    assert _children_cache_size(report) == 1

    display_2 = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test"
    )
    assert display_1.importances.equals(display_2.importances)
    assert _children_cache_size(report) == 1


def test_cache_seed_none(comparison_cv_report_ridge):
    report = comparison_cv_report_ridge
    assert _children_cache_size(report) == 0

    report.inspection.permutation_importance(n_repeats=2, data_source="test")
    assert _children_cache_size(report) == 1

    report.inspection.permutation_importance(n_repeats=2, data_source="test")
    assert _children_cache_size(report) == 1


def test_cache_parameter_in_cache(comparison_cv_report_ridge):
    report = comparison_cv_report_ridge
    report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test", metric="r2"
    )
    child_report = next(iter(report.reports_.values())).estimator_reports_[0]
    with check_cache_changed(child_report._cache):
        report.inspection.permutation_importance(
            seed=42,
            n_repeats=2,
            data_source="test",
            metric=make_scorer(root_mean_squared_error),
        )


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(comparison_cv_report_ridge, data_source):
    report = comparison_cv_report_ridge
    display = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source=data_source
    )
    assert set(display.importances["data_source"]) == {data_source}


def test_seed_wrong_type(comparison_cv_report_ridge):
    report = comparison_cv_report_ridge
    with pytest.raises(
        ValueError, match="seed must be an integer or None; got <class 'str'>"
    ):
        report.inspection.permutation_importance(seed="42")

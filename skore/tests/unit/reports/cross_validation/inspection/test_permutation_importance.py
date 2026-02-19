import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skore import CrossValidationReport, PermutationImportanceDisplay
from skore._utils._testing import check_cache_changed


@pytest.mark.parametrize(
    "data_fixture, estimator",
    [
        ("regression_data", Ridge()),
        ("regression_data", make_pipeline(StandardScaler(), Ridge())),
        (
            "binary_classification_data",
            LogisticRegression(),
        ),
        (
            "binary_classification_data",
            make_pipeline(StandardScaler(), LogisticRegression()),
        ),
    ],
)
def test_returns_display(data_fixture, estimator, request):
    X, y = request.getfixturevalue(data_fixture)
    report = CrossValidationReport(estimator, X, y, splitter=2)
    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    assert isinstance(display, PermutationImportanceDisplay)


def test_at_step(request):
    X, y = request.getfixturevalue("regression_data")
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), Ridge())
    report = CrossValidationReport(pipeline, X, y, splitter=2)

    display_raw = report.inspection.permutation_importance(
        seed=42, n_repeats=2, at_step=0
    )
    display_pca = report.inspection.permutation_importance(
        seed=42, n_repeats=2, at_step=-1
    )
    raw_features = set(display_raw.importances["feature"].unique())
    pca_features = set(display_pca.importances["feature"].unique())
    assert raw_features != pca_features
    assert len(raw_features) == 4
    assert len(pca_features) == 2


def test_cache_seed_int(request):
    X, y = request.getfixturevalue("regression_data")
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    assert report._cache == {}

    display1 = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test"
    )
    assert len(report._cache) == 1

    display2 = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="test"
    )
    assert display1.importances.equals(display2.importances)
    assert len(report._cache) == 1


def test_cache_seed_none(request):
    X, y = request.getfixturevalue("regression_data")
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    assert report._cache == {}

    report.inspection.permutation_importance(n_repeats=2, data_source="test")
    assert len(report._cache) == 1

    display2 = report.inspection.permutation_importance(n_repeats=2, data_source="test")
    assert len(report._cache) == 1
    cached_display = next(iter(report._cache.values()))
    assert cached_display is display2


def test_cache_parameter_in_cache(request):
    X, y = request.getfixturevalue("regression_data")
    report = CrossValidationReport(Ridge(), X, y, splitter=2)

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


def test_split_column(request):
    X, y = request.getfixturevalue("regression_data")
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.inspection.permutation_importance(seed=42, n_repeats=2)
    df = display.importances
    assert set(df["split"].unique()) == {0, 1}


@pytest.mark.parametrize("data_source", ["train", "test"])
def test_data_source(request, data_source):
    X, y = request.getfixturevalue("regression_data")
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source=data_source
    )
    assert display.importances["data_source"].unique() == [data_source]


def test_data_source_X_y(regression_data):
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.inspection.permutation_importance(
        seed=42, n_repeats=2, data_source="X_y", X=X, y=y
    )
    assert isinstance(display, PermutationImportanceDisplay)
    assert display.importances["data_source"].unique() == ["X_y"]


def test_seed_wrong_type(regression_data):
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    with pytest.raises(
        ValueError, match="seed must be an integer or None; got <class 'str'>"
    ):
        report.inspection.permutation_importance(seed="42")


def test_no_target(regression_data):
    X, _ = regression_data
    report = CrossValidationReport(KMeans(), X, splitter=2)
    with pytest.raises(
        ValueError,
        match="Permutation importance can not be performed on a clustering model.",
    ):
        report.inspection.permutation_importance(seed=42)

    with pytest.raises(
        ValueError,
        match="Permutation importance can not be performed on a clustering model.",
    ):
        report.inspection.permutation_importance(
            seed=42, data_source="X_y", X=X, y=None
        )

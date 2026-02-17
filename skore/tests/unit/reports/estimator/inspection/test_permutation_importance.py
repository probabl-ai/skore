import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from skore import EstimatorReport, PermutationImportanceDisplay
from skore._utils._testing import check_cache_changed


def test_not_fitted_error(regression_train_test_split):
    _, X_test, _, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(), fit=False, X_test=X_test, y_test=y_test
    )
    with pytest.raises(NotFittedError, match="This LinearRegression instance"):
        report.inspection.permutation_importance(seed=42)


@pytest.mark.parametrize(
    "at_step, expected_features",
    [
        (0, ["Feature #0", "Feature #1", "Feature #2", "Feature #3"]),
        (1, ["x0", "x1", "x2", "x3"]),
        (-1, ["pca0", "pca1"]),
    ],
)
def test_at_step_int(regression_train_test_split, at_step, expected_features):
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(seed=42, at_step=at_step)
    assert isinstance(display, PermutationImportanceDisplay)
    actual_features = sorted(display.importances["feature"].unique())
    assert actual_features == sorted(expected_features)


def test_at_step_str(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(seed=42, at_step="pca")
    assert isinstance(display, PermutationImportanceDisplay)
    actual_features = sorted(display.importances["feature"].unique())
    assert actual_features == sorted(["x0", "x1", "x2", "x3"])


def test_at_step_non_pipeline_int(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display_start = report.inspection.permutation_importance(seed=42, at_step=0)
    display_end = report.inspection.permutation_importance(seed=42, at_step=-1)
    assert len(display_start.importances) == len(display_end.importances)


def test_at_step_non_pipeline_str(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    with pytest.raises(
        ValueError,
        match="at_step can only be a string when the estimator is a Pipeline",
    ):
        report.inspection.permutation_importance(seed=42, at_step="StandardScaler")


@pytest.mark.parametrize(
    "at_step, err_msg",
    [
        (8, "at_step must be strictly smaller in magnitude than"),
        (-8, "at_step must be strictly smaller in magnitude than"),
        ("hello", "not in list"),
        (0.5, "at_step must be an integer or a string"),
    ],
)
def test_at_step_value_error(regression_train_test_split, at_step, err_msg):
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    with pytest.raises(ValueError, match=err_msg):
        report.inspection.permutation_importance(seed=42, at_step=at_step)


def test_cache_display_stored(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert report._cache == {}

    with check_cache_changed(report._cache):
        display = report.inspection.permutation_importance(data_source="train", seed=42)

    assert len(report._cache) == 1
    cached_display = next(iter(report._cache.values()))
    assert isinstance(cached_display, PermutationImportanceDisplay)
    assert cached_display is display


@pytest.mark.parametrize(
    "param_name, first_value, second_value, use_pipeline",
    [
        ("metrics", "r2", make_scorer(root_mean_squared_error), False),
        ("at_step", 0, -1, True),
        ("max_samples", 0.5, 0.8, False),
    ],
)
def test_cache_parameter_in_cache(
    regression_train_test_split, param_name, first_value, second_value, use_pipeline
):
    X_train, X_test, y_train, y_test = regression_train_test_split

    if use_pipeline:
        estimator = make_pipeline(
            StandardScaler(), PCA(n_components=2), LinearRegression()
        )
    else:
        estimator = LinearRegression()

    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    kwargs = {"data_source": "train", "seed": 42, param_name: first_value}
    report.inspection.permutation_importance(**kwargs)

    with check_cache_changed(report._cache):
        kwargs[param_name] = second_value
        report.inspection.permutation_importance(**kwargs)


def test_cache_seed_none(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert report._cache == {}

    report.inspection.permutation_importance(data_source="train")
    assert len(report._cache) == 1

    display2 = report.inspection.permutation_importance(data_source="train")
    assert len(report._cache) == 1
    cached_display = next(iter(report._cache.values()))
    assert cached_display is display2


def test_cache_seed_int(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert report._cache == {}

    display1 = report.inspection.permutation_importance(data_source="train", seed=42)
    assert len(report._cache) == 1

    display2 = report.inspection.permutation_importance(data_source="train", seed=42)
    assert display1.importances.equals(display2.importances)
    assert len(report._cache) == 1


def test_sparse_array(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(SplineTransformer(sparse_output=True), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(seed=42, at_step=-1)
    assert isinstance(display, PermutationImportanceDisplay)
    assert len(display.importances) > 0


def test_feature_names_from_pipeline(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(SplineTransformer(), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.inspection.permutation_importance(seed=42, at_step=-1)
    assert isinstance(display, PermutationImportanceDisplay)
    assert "feature" in display.importances.columns
    assert len(display.importances["feature"].unique()) > 0


def test_seed_wrong_type(regression_train_test_split):
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    with pytest.raises(
        ValueError, match="seed must be an integer or None; got <class 'str'>"
    ):
        report.inspection.permutation_importance(seed="42")


def test_no_target(regression_train_test_split):
    X_train, X_test, _, _ = regression_train_test_split
    report = EstimatorReport(
        KMeans(),
        X_train=X_train,
        X_test=X_test,
    )
    with pytest.raises(
        ValueError,
        match="Permutation importance can not be performed on a clustering model.",
    ):
        report.inspection.permutation_importance(seed=42)

import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from skore import EstimatorReport, PermutationImportanceDisplay
from skore._utils._testing import check_cache_changed


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_permutation_importance_returns_display(regression_train_test_split, n_jobs):
    """Test that permutation importance returns a PermutationImportanceDisplay."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.inspection.permutation_importance(seed=42, n_jobs=n_jobs)

    assert isinstance(display, PermutationImportanceDisplay)
    assert hasattr(display, "importances")
    assert hasattr(display, "frame")


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_not_fitted_error(regression_train_test_split, n_jobs):
    """Test that NotFittedError is raised when estimator is not fitted."""
    _, X_test, _, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(), fit=False, X_test=X_test, y_test=y_test
    )

    error_msg = "This LinearRegression instance is not fitted yet"
    with pytest.raises(NotFittedError, match=error_msg):
        report.inspection.permutation_importance(seed=42, n_jobs=n_jobs)


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    "at_step, expected_features",
    [
        (0, ["Feature #0", "Feature #1", "Feature #2", "Feature #3"]),
        (1, ["x0", "x1", "x2", "x3"]),
        (-1, ["pca0", "pca1"]),
    ],
)
def test_at_step_int(regression_train_test_split, n_jobs, at_step, expected_features):
    """Test the `at_step` integer parameter for permutation importance with a
    pipeline."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.permutation_importance(
        seed=42, at_step=at_step, n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    actual_features = sorted(display.importances["feature"].unique())
    assert actual_features == sorted(expected_features)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_at_step_str(regression_train_test_split, n_jobs):
    """Test the `at_step` string parameter for permutation importance with a
    pipeline."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.permutation_importance(
        seed=42, at_step="pca", n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    expected_features = ["x0", "x1", "x2", "x3"]
    actual_features = sorted(display.importances["feature"].unique())
    assert actual_features == sorted(expected_features)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_at_step_non_pipeline(regression_train_test_split, n_jobs):
    """For non-pipeline estimators, changing at_step should not change the results."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display_start = report.inspection.permutation_importance(
        seed=42, at_step=0, n_jobs=n_jobs
    )
    display_end = report.inspection.permutation_importance(
        seed=42, at_step=-1, n_jobs=n_jobs
    )
    assert len(display_start.importances) == len(display_end.importances)


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    "at_step, err_msg",
    [
        (8, "at_step must be strictly smaller in magnitude than"),
        (-8, "at_step must be strictly smaller in magnitude than"),
        ("hello", "not in list"),
        (0.5, "at_step must be an integer or a string"),
    ],
)
def test_at_step_value_error(regression_train_test_split, n_jobs, at_step, err_msg):
    """If `at_step` value is not appropriate, a ValueError is raised."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    with pytest.raises(ValueError, match=err_msg):
        report.inspection.permutation_importance(
            seed=42, at_step=at_step, n_jobs=n_jobs
        )


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    "metric, expected_metric_name",
    [
        ("r2", "r2"),
        (make_scorer(r2_score), "r2 score"),
        (make_scorer(root_mean_squared_error), "root mean squared error"),
    ],
)
def test_metric_parameter(
    regression_train_test_split, n_jobs, metric, expected_metric_name
):
    """Test that different metric parameters work correctly."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display = report.inspection.permutation_importance(
        seed=42, metric=metric, n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    metrics = display.importances["metric"].unique()
    assert len(metrics) == 1
    assert metrics[0] == expected_metric_name


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_metric_list(regression_train_test_split, n_jobs):
    """Test that metric as a list works correctly."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display = report.inspection.permutation_importance(
        seed=42, metric=["r2", "neg_mean_squared_error"], n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    metrics = display.importances["metric"].unique()
    assert len(metrics) == 2
    assert set(metrics) == {"r2", "neg_mean_squared_error"}


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_metric_dict(regression_train_test_split, n_jobs):
    """Test that metric as a dict works correctly."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    metric_dict = {
        "r2": make_scorer(r2_score),
        "rmse": make_scorer(root_mean_squared_error),
    }
    display = report.inspection.permutation_importance(
        seed=42, metric=metric_dict, n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    # Check that we have the expected unique metric values (dict keys)
    metrics = display.importances["metric"].unique()
    assert len(metrics) == 2
    assert set(metrics) == {"r2", "rmse"}


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("max_samples", [0.5, 1.0])
def test_max_samples_parameter(regression_train_test_split, n_jobs, max_samples):
    """Test that max_samples parameter works correctly."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display = report.inspection.permutation_importance(
        seed=42, max_samples=max_samples, n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    assert len(display.importances) > 0


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cache_display_stored(regression_train_test_split, n_jobs):
    """Test that the display object is stored in cache."""
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
        display = report.inspection.permutation_importance(
            data_source="train", seed=42, n_jobs=n_jobs
        )

    assert len(report._cache) == 1
    key = next(iter(report._cache.keys()))
    cached_display = report._cache[key]
    assert isinstance(cached_display, PermutationImportanceDisplay)
    assert cached_display is display


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    "param_name, first_value, second_value, use_pipeline",
    [
        ("metric", "r2", make_scorer(root_mean_squared_error), False),
        ("at_step", 0, -1, True),
        ("max_samples", 0.5, 0.8, False),
    ],
)
def test_cache_parameter_in_cache(
    regression_train_test_split,
    n_jobs,
    param_name,
    first_value,
    second_value,
    use_pipeline,
):
    """Test that metric, at_step, and max_samples are part of the cache key."""
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

    # First call with first parameter value
    kwargs1 = {
        "data_source": "train",
        "seed": 42,
        "n_jobs": n_jobs,
        param_name: first_value,
    }
    report.inspection.permutation_importance(**kwargs1)

    # Different parameter value should create a new cache entry
    with check_cache_changed(report._cache):
        kwargs2 = {
            "data_source": "train",
            "seed": 42,
            "n_jobs": n_jobs,
            param_name: second_value,
        }
        report.inspection.permutation_importance(**kwargs2)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cache_seed_none(regression_train_test_split, n_jobs):
    """Test cache behavior when seed is None - should not cache."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert report._cache == {}

    report.inspection.permutation_importance(data_source="train", n_jobs=n_jobs)
    assert len(report._cache) == 1

    display2 = report.inspection.permutation_importance(
        data_source="train", n_jobs=n_jobs
    )
    assert len(report._cache) == 1
    key = next(iter(report._cache.keys()))
    cached_display = report._cache[key]
    assert cached_display is display2


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cache_seed_int(regression_train_test_split, n_jobs):
    """Test cache behavior when seed is an int - should use cache."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        LinearRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert report._cache == {}

    display1 = report.inspection.permutation_importance(
        data_source="train", seed=42, n_jobs=n_jobs
    )
    assert display1 is not None
    assert isinstance(display1, PermutationImportanceDisplay)
    assert len(report._cache) == 1

    display2 = report.inspection.permutation_importance(
        data_source="train", seed=42, n_jobs=n_jobs
    )
    assert display2 is not None
    assert isinstance(display2, PermutationImportanceDisplay)
    assert display1.importances.equals(display2.importances)
    assert len(report._cache) == 1


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_classification(logistic_binary_classification_with_train_test, n_jobs):
    """Test permutation importance with classification data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.permutation_importance(
        data_source="train", seed=42, n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    assert len(display.importances) > 0
    assert "accuracy" in display.importances["metric"].unique()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_sparse_array(regression_train_test_split, n_jobs):
    """Test that permutation importance works with sparse arrays."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(SplineTransformer(sparse_output=True), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.permutation_importance(
        seed=42, at_step=-1, n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    assert len(display.importances) > 0


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_feature_names_from_pipeline(regression_train_test_split, n_jobs):
    """Test that feature names from pipeline steps are used correctly."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    pipeline = make_pipeline(SplineTransformer(), LinearRegression())
    report = EstimatorReport(
        pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    display = report.inspection.permutation_importance(
        seed=42, at_step=-1, n_jobs=n_jobs
    )

    assert isinstance(display, PermutationImportanceDisplay)
    assert "feature" in display.importances.columns
    assert len(display.importances["feature"].unique()) > 0

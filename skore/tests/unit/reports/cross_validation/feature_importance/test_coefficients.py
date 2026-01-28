import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from skore import CoefficientsDisplay, CrossValidationReport


@pytest.mark.parametrize(
    "estimator",
    [
        Ridge(),
        TransformedTargetRegressor(Ridge()),
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("transformed_ridge", TransformedTargetRegressor(Ridge())),
            ]
        ),
    ],
)
def test_with_model_exposing_coef(regression_data, estimator):
    """Check that we can create a coefficients display from model exposing a `coef_`
    attribute."""
    X, y = regression_data
    report = CrossValidationReport(estimator, X, y, splitter=2)
    assert hasattr(report.feature_importance, "coefficients")
    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)


def test_with_model_not_exposing_coef(regression_data):
    """Check that we cannot create a coefficients display from model not exposing a
    `coef_` attribute."""
    X, y = regression_data
    report = CrossValidationReport(DecisionTreeRegressor(), X, y, splitter=2)
    assert not hasattr(report.feature_importance, "coefficients")


def test_select_k_zero_raises_error(regression_data):
    """Test that select_k=0 raises ValueError."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    with pytest.raises(ValueError, match="`select_k` must be a non-zero integer"):
        display.frame(select_k=0)


def test_select_k_positive_frame(regression_data):
    """Test that select_k with positive value returns correct features in frame."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    select_k = min(3, full_frame["feature"].nunique())

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=select_k)

    # Calculate expected features based on mean absolute coefficients across splits
    mean_abs_coefs = full_frame.groupby("feature")["coefficients"].apply(
        lambda x: x.abs().mean()
    )
    expected_features = set(mean_abs_coefs.nlargest(select_k).index.tolist())

    # Verify the filtered frame contains exactly the expected features
    assert set(filtered_frame["feature"].unique()) == expected_features
    # For CV, we have select_k features × n_splits rows
    n_splits = full_frame["split"].nunique()
    assert len(filtered_frame) == select_k * n_splits


def test_select_k_negative_frame(regression_data):
    """Test that select_k with negative value returns correct features in frame."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    bottom_k = min(3, full_frame["feature"].nunique())

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=-bottom_k)

    # Calculate expected features based on mean absolute coefficients across splits
    mean_abs_coefs = full_frame.groupby("feature")["coefficients"].apply(
        lambda x: x.abs().mean()
    )
    expected_features = set(mean_abs_coefs.nsmallest(bottom_k).index.tolist())

    # Verify the filtered frame contains exactly the expected features
    assert set(filtered_frame["feature"].unique()) == expected_features
    # For CV, we have bottom_k features × n_splits rows
    n_splits = full_frame["split"].nunique()
    assert len(filtered_frame) == bottom_k * n_splits


def test_select_k_multiclass(regression_data):
    """Test that select_k works correctly per class in
    multiclass classification with CV."""
    from sklearn.datasets import load_iris

    # iris dataset - has 3 classes
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    report = CrossValidationReport(
        LogisticRegression(max_iter=200),
        X,
        y,
        splitter=2,
    )
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    select_k = 2

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=select_k)

    # For each label, verify we have exactly select_k features (across all splits)
    for label in filtered_frame["label"].unique():
        label_features = filtered_frame[filtered_frame["label"] == label]
        n_splits = label_features["split"].nunique()
        # Should have select_k features × n_splits rows
        assert len(label_features) == select_k * n_splits, (
            f"Expected {select_k * n_splits} rows for label {label}"
        )
        # Should have exactly select_k unique features
        assert label_features["feature"].nunique() == select_k, (
            f"Expected {select_k} unique features for label {label}"
        )

        # Verify these are indeed the top-k for this label based on mean across splits
        label_full = full_frame[full_frame["label"] == label]
        mean_abs_coefs = label_full.groupby("feature")["coefficients"].apply(
            lambda x: x.abs().mean()
        )
        expected_features = set(mean_abs_coefs.nlargest(select_k).index.tolist())
        actual_features = set(label_features["feature"].unique())
        assert actual_features == expected_features, (
            f"For label {label}, expected features {expected_features}, "
            f"but got {actual_features}"
        )


def test_sorting_order_descending(regression_data):
    """Test that sorting_order='descending' sorts by
    largest absolute coefficients first."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="descending")

    # For CV, features are sorted by mean absolute coefficient across splits
    # Group by feature and check that means are in descending order
    mean_abs_coefs = sorted_frame.groupby("feature", sort=False)["coefficients"].apply(
        lambda x: x.abs().mean()
    )
    assert np.all(mean_abs_coefs.values[:-1] >= mean_abs_coefs.values[1:])


def test_sorting_order_ascending(regression_data):
    """Test that sorting_order='ascending' sorts by
    smallest absolute coefficients first."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="ascending")

    # For CV, features are sorted by mean absolute coefficient across splits
    # Group by feature and check that means are in ascending order
    mean_abs_coefs = sorted_frame.groupby("feature", sort=False)["coefficients"].apply(
        lambda x: x.abs().mean()
    )
    assert np.all(mean_abs_coefs.values[:-1] <= mean_abs_coefs.values[1:])


def test_select_k_with_sorting_descending(regression_data):
    """Test that select_k and sorting_order work together correctly."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    select_k = 3
    filtered_sorted_frame = display.frame(
        include_intercept=False, select_k=select_k, sorting_order="descending"
    )

    # Should have exactly select_k features × n_splits rows
    n_splits = filtered_sorted_frame["split"].nunique()
    assert filtered_sorted_frame["feature"].nunique() == select_k
    assert len(filtered_sorted_frame) == select_k * n_splits

    # Should be sorted in descending order by mean absolute coefficient
    mean_abs_coefs = filtered_sorted_frame.groupby("feature", sort=False)[
        "coefficients"
    ].apply(lambda x: x.abs().mean())
    assert np.all(mean_abs_coefs.values[:-1] >= mean_abs_coefs.values[1:])


def test_select_k_negative_with_sorting_ascending(regression_data):
    """Test that negative select_k with ascending sort works correctly."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    bottom_k = 3
    filtered_sorted_frame = display.frame(
        include_intercept=False, select_k=-bottom_k, sorting_order="ascending"
    )

    # Should have exactly bottom_k features × n_splits rows
    n_splits = filtered_sorted_frame["split"].nunique()
    assert filtered_sorted_frame["feature"].nunique() == bottom_k
    assert len(filtered_sorted_frame) == bottom_k * n_splits

    # Should be sorted in ascending order by mean absolute coefficient
    mean_abs_coefs = filtered_sorted_frame.groupby("feature", sort=False)[
        "coefficients"
    ].apply(lambda x: x.abs().mean())
    assert np.all(mean_abs_coefs.values[:-1] <= mean_abs_coefs.values[1:])


def test_plot_with_select_k(regression_data):
    """Test that plot method correctly uses select_k parameter."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    select_k = 3

    # Should not raise an error
    display.plot(select_k=select_k, include_intercept=False)

    # Verify the plot was created
    assert display.ax_ is not None
    assert display.figure_ is not None

    # Verify correct number of features plotted
    ax = (
        display.ax_
        if not isinstance(display.ax_, np.ndarray)
        else display.ax_.flatten()[0]
    )
    plotted_features = {label.get_text() for label in ax.get_yticklabels()}
    assert len(plotted_features) == select_k


def test_plot_with_sorting_order(regression_data):
    """Test that plot method correctly uses sorting_order parameter."""
    X, y = regression_data
    report = CrossValidationReport(Ridge(), X, y, splitter=2)
    display = report.feature_importance.coefficients()

    # Should not raise an error
    display.plot(sorting_order="ascending", include_intercept=False)

    # Verify the plot was created
    assert display.ax_ is not None
    assert display.figure_ is not None

import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from skore import CoefficientsDisplay, EstimatorReport


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
def test_with_model_exposing_coef(regression_train_test_split, estimator):
    """Check that we can create a coefficients display from model exposing a `coef_`
    attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    assert hasattr(report.feature_importance, "coefficients")
    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)


def test_with_model_not_exposing_coef(regression_train_test_split):
    """Check that we cannot create a coefficients display from model not exposing a
    `coef_` attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        DecisionTreeRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    assert not hasattr(report.feature_importance, "coefficients")


def test_select_k_zero_raises_error(regression_train_test_split):
    """Test that select_k=0 raises ValueError."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.feature_importance.coefficients()

    with pytest.raises(ValueError, match="`select_k` must be a non-zero integer"):
        display.frame(select_k=0)


def test_select_k_positive_frame(regression_train_test_split):
    """Test that select_k with positive value returns correct features in frame."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    select_k = min(3, full_frame["feature"].nunique())

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=select_k)

    # Calculate expected features
    abs_coefs = full_frame["coefficients"].abs()
    expected_features = set(
        full_frame.loc[abs_coefs.nlargest(select_k).index, "feature"].tolist()
    )

    # Verify the filtered frame contains exactly the expected features
    assert set(filtered_frame["feature"].unique()) == expected_features
    assert len(filtered_frame) == select_k


def test_select_k_negative_frame(regression_train_test_split):
    """Test that select_k with negative value returns correct features in frame."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    bottom_k = min(3, full_frame["feature"].nunique())

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=-bottom_k)

    # Calculate expected features
    abs_coefs = full_frame["coefficients"].abs()
    expected_features = set(
        full_frame.loc[abs_coefs.nsmallest(bottom_k).index, "feature"].tolist()
    )

    # Verify the filtered frame contains exactly the expected features
    assert set(filtered_frame["feature"].unique()) == expected_features
    assert len(filtered_frame) == bottom_k


def test_select_k_multiclass():
    """Test that select_k works correctly per class in multiclass classification."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # iris dataset - has 3 classes
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    report = EstimatorReport(
        LogisticRegression(max_iter=200),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    select_k = 2

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=select_k)

    # For each label, verify we have exactly select_k features
    for label in filtered_frame["label"].unique():
        label_features = filtered_frame[filtered_frame["label"] == label]
        assert len(label_features) == select_k, (
            f"Expected {select_k} features for label {label}"
        )

        # Verify these are indeed the top-k for this label
        label_full = full_frame[full_frame["label"] == label]
        abs_coefs = label_full["coefficients"].abs()
        expected_features = set(
            label_full.loc[abs_coefs.nlargest(select_k).index, "feature"].tolist()
        )
        actual_features = set(label_features["feature"].unique())
        assert actual_features == expected_features, (
            f"For label {label}, expected features {expected_features}, "
            f"but got {actual_features}"
        )


def test_sorting_order_descending(regression_train_test_split):
    """Test that sorting_order='descending' sorts by
    largest absolute coefficients first."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="descending")

    # Verify features are sorted by descending absolute coefficient
    abs_coefs = sorted_frame["coefficients"].abs().values
    assert np.all(abs_coefs[:-1] >= abs_coefs[1:])


def test_sorting_order_ascending(regression_train_test_split):
    """Test that sorting_order='ascending' sorts by
    smallest absolute coefficients first."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="ascending")

    # Verify features are sorted by ascending absolute coefficient
    abs_coefs = sorted_frame["coefficients"].abs().values
    assert np.all(abs_coefs[:-1] <= abs_coefs[1:])


def test_select_k_with_sorting_descending(regression_train_test_split):
    """Test that select_k and sorting_order work together correctly."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.feature_importance.coefficients()

    select_k = 3
    filtered_sorted_frame = display.frame(
        include_intercept=False, select_k=select_k, sorting_order="descending"
    )

    # Should have exactly select_k features
    assert len(filtered_sorted_frame) == select_k

    # Should be sorted in descending order
    abs_coefs = filtered_sorted_frame["coefficients"].abs().values
    assert np.all(abs_coefs[:-1] >= abs_coefs[1:])


def test_select_k_negative_with_sorting_ascending(regression_train_test_split):
    """Test that negative select_k with ascending sort works correctly."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.feature_importance.coefficients()

    bottom_k = 3
    filtered_sorted_frame = display.frame(
        include_intercept=False, select_k=-bottom_k, sorting_order="ascending"
    )

    # Should have exactly bottom_k features
    assert len(filtered_sorted_frame) == bottom_k

    # Should be sorted in ascending order
    abs_coefs = filtered_sorted_frame["coefficients"].abs().values
    assert np.all(abs_coefs[:-1] <= abs_coefs[1:])


def test_plot_with_select_k(regression_train_test_split):
    """Test that plot method correctly uses select_k parameter."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
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


def test_plot_with_sorting_order(regression_train_test_split):
    """Test that plot method correctly uses sorting_order parameter."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.feature_importance.coefficients()

    # Should not raise an error
    display.plot(sorting_order="ascending", include_intercept=False)

    # Verify the plot was created
    assert display.ax_ is not None
    assert display.figure_ is not None

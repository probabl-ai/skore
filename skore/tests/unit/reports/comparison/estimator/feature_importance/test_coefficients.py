import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from skore import CoefficientsDisplay, ComparisonReport, EstimatorReport


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
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.feature_importance, "coefficients")
    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)


def test_with_model_not_exposing_coef(regression_train_test_split):
    """Check that we cannot create a coefficients display from model not exposing a
    `coef_` attribute."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        DecisionTreeRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        DecisionTreeRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.feature_importance, "coefficients")


def test_with_mixed_reports(regression_train_test_split):
    """Check that we cannot create a coefficients display from mixed reports."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        DecisionTreeRegressor(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.feature_importance, "coefficients")


def test_select_k_positive_frame(regression_train_test_split):
    """Test that select_k with positive value selects features per estimator."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    select_k = min(3, full_frame["feature"].nunique())

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=select_k)

    # For comparison reports, selection happens per estimator
    for estimator_name in filtered_frame["estimator"].unique():
        est_features = filtered_frame[filtered_frame["estimator"] == estimator_name]

        # Should have exactly select_k features for this estimator
        assert len(est_features) == select_k, (
            f"Expected {select_k} features for {estimator_name}"
        )

        # Verify these are the top-k for this estimator
        est_full = full_frame[full_frame["estimator"] == estimator_name]
        abs_coefs = est_full["coefficients"].abs()
        expected_features = set(
            est_full.loc[abs_coefs.nlargest(select_k).index, "feature"].tolist()
        )
        actual_features = set(est_features["feature"].unique())
        assert actual_features == expected_features, (
            f"For {estimator_name}, expected {expected_features}, got {actual_features}"
        )


def test_select_k_negative_frame(regression_train_test_split):
    """Test that select_k with negative value selects features per estimator."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    bottom_k = min(3, full_frame["feature"].nunique())

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=-bottom_k)

    # For comparison reports, selection happens per estimator
    for estimator_name in filtered_frame["estimator"].unique():
        est_features = filtered_frame[filtered_frame["estimator"] == estimator_name]

        # Should have exactly bottom_k features for this estimator
        assert len(est_features) == bottom_k, (
            f"Expected {bottom_k} features for {estimator_name}"
        )

        # Verify these are the bottom-k for this estimator
        est_full = full_frame[full_frame["estimator"] == estimator_name]
        abs_coefs = est_full["coefficients"].abs()
        expected_features = set(
            est_full.loc[abs_coefs.nsmallest(bottom_k).index, "feature"].tolist()
        )
        actual_features = set(est_features["feature"].unique())
        assert actual_features == expected_features, (
            f"For {estimator_name}, expected {expected_features}, got {actual_features}"
        )


def test_select_k_multiclass(regression_train_test_split):
    """Test that select_k works per estimator and per class in multiclass comparison."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # iris dataset - has 3 classes
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    report_1 = EstimatorReport(
        LogisticRegression(max_iter=200),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        LogisticRegression(max_iter=200),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    select_k = 2

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=select_k)

    # For each estimator and label combination
    for estimator_name in filtered_frame["estimator"].unique():
        for label in filtered_frame["label"].unique():
            group_filtered = filtered_frame[
                (filtered_frame["estimator"] == estimator_name)
                & (filtered_frame["label"] == label)
            ]

            # Should have exactly select_k features
            assert len(group_filtered) == select_k, (
                f"Expected {select_k} features for {estimator_name}, label {label}"
            )

            # Verify these are the top-k for this estimator-label combination
            group_full = full_frame[
                (full_frame["estimator"] == estimator_name)
                & (full_frame["label"] == label)
            ]
            abs_coefs = group_full["coefficients"].abs()
            expected_features = set(
                group_full.loc[abs_coefs.nlargest(select_k).index, "feature"].tolist()
            )
            actual_features = set(group_filtered["feature"].unique())
            assert actual_features == expected_features, (
                f"For {estimator_name}, label {label}, expected {expected_features}, "
                f"got {actual_features}"
            )


def test_sorting_order_descending(regression_train_test_split):
    """Test that sorting_order='descending' sorts per estimator."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="descending")

    # For each estimator, verify sorting
    for estimator_name in sorted_frame["estimator"].unique():
        est_frame = sorted_frame[sorted_frame["estimator"] == estimator_name]
        abs_coefs = est_frame["coefficients"].abs().values
        assert np.all(abs_coefs[:-1] >= abs_coefs[1:]), (
            f"Features not sorted in descending order for {estimator_name}"
        )


def test_sorting_order_ascending(regression_train_test_split):
    """Test that sorting_order='ascending' sorts per estimator."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="ascending")

    # For each estimator, verify sorting
    for estimator_name in sorted_frame["estimator"].unique():
        est_frame = sorted_frame[sorted_frame["estimator"] == estimator_name]
        abs_coefs = est_frame["coefficients"].abs().values
        assert np.all(abs_coefs[:-1] <= abs_coefs[1:]), (
            f"Features not sorted in ascending order for {estimator_name}"
        )


def test_plot_with_select_k(regression_train_test_split):
    """Test that plot method correctly uses select_k parameter."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    select_k = 3

    # Should not raise an error
    display.plot(select_k=select_k, include_intercept=False)

    # Verify the plot was created
    assert display.ax_ is not None
    assert display.figure_ is not None


def test_plot_with_sorting_order(regression_train_test_split):
    """Test that plot method correctly uses sorting_order parameter."""
    X_train, X_test, y_train, y_test = regression_train_test_split
    report_1 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        Ridge(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    # Should not raise an error
    display.plot(sorting_order="ascending", include_intercept=False)

    # Verify the plot was created
    assert display.ax_ is not None
    assert display.figure_ is not None

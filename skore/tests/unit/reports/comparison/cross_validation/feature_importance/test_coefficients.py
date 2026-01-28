import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from skore import CoefficientsDisplay, ComparisonReport, CrossValidationReport


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
    report_1 = CrossValidationReport(estimator, X, y, splitter=2)
    report_2 = CrossValidationReport(estimator, X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert hasattr(report.feature_importance, "coefficients")
    display = report.feature_importance.coefficients()
    assert isinstance(display, CoefficientsDisplay)


def test_with_model_not_exposing_coef(regression_data):
    """Check that we cannot create a coefficients display from model not exposing a
    `coef_` attribute."""
    X, y = regression_data
    report_1 = CrossValidationReport(DecisionTreeRegressor(), X, y, splitter=2)
    report_2 = CrossValidationReport(DecisionTreeRegressor(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.feature_importance, "coefficients")


def test_with_mixed_reports(regression_data):
    """Check that we cannot create a coefficients display from mixed reports."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(DecisionTreeRegressor(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    assert not hasattr(report.feature_importance, "coefficients")


def test_select_k_zero_raises_error(regression_data):
    """Test that select_k=0 raises ValueError."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    with pytest.raises(ValueError, match="`select_k` must be a non-zero integer"):
        display.frame(select_k=0)


def test_select_k_positive_frame(regression_data):
    """Test that select_k selects features per estimator based on mean across splits."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    select_k = min(3, full_frame["feature"].nunique())

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=select_k)

    # For each estimator, verify selection based on mean across splits
    for estimator_name in filtered_frame["estimator"].unique():
        est_filtered = filtered_frame[filtered_frame["estimator"] == estimator_name]

        # Should have select_k features × n_splits rows
        n_splits = est_filtered["split"].nunique()
        assert est_filtered["feature"].nunique() == select_k, (
            f"Expected {select_k} unique features for {estimator_name}"
        )
        assert len(est_filtered) == select_k * n_splits, (
            f"Expected {select_k * n_splits} rows for {estimator_name}"
        )

        # Verify these are the top-k for this estimator based on mean
        est_full = full_frame[full_frame["estimator"] == estimator_name]
        mean_abs_coefs = est_full.groupby("feature")["coefficients"].apply(
            lambda x: x.abs().mean()
        )
        expected_features = set(mean_abs_coefs.nlargest(select_k).index.tolist())
        actual_features = set(est_filtered["feature"].unique())
        assert actual_features == expected_features, (
            f"For {estimator_name}, expected {expected_features}, got {actual_features}"
        )


def test_select_k_negative_frame(regression_data):
    """Test that negative select_k selects features per estimator based on
    mean across splits."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    bottom_k = min(3, full_frame["feature"].nunique())

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=-bottom_k)

    # For each estimator, verify selection based on mean across splits
    for estimator_name in filtered_frame["estimator"].unique():
        est_filtered = filtered_frame[filtered_frame["estimator"] == estimator_name]

        # Should have bottom_k features × n_splits rows
        n_splits = est_filtered["split"].nunique()
        assert est_filtered["feature"].nunique() == bottom_k, (
            f"Expected {bottom_k} unique features for {estimator_name}"
        )
        assert len(est_filtered) == bottom_k * n_splits, (
            f"Expected {bottom_k * n_splits} rows for {estimator_name}"
        )

        # Verify these are the bottom-k for this estimator based on mean
        est_full = full_frame[full_frame["estimator"] == estimator_name]
        mean_abs_coefs = est_full.groupby("feature")["coefficients"].apply(
            lambda x: x.abs().mean()
        )
        expected_features = set(mean_abs_coefs.nsmallest(bottom_k).index.tolist())
        actual_features = set(est_filtered["feature"].unique())
        assert actual_features == expected_features, (
            f"For {estimator_name}, expected {expected_features}, got {actual_features}"
        )


def test_select_k_multiclass(regression_data):
    """Test that select_k works per estimator and per class with CV."""
    from sklearn.datasets import load_iris

    # iris dataset - has 3 classes
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    report_1 = CrossValidationReport(LogisticRegression(max_iter=200), X, y, splitter=2)
    report_2 = CrossValidationReport(LogisticRegression(max_iter=200), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    full_frame = display.frame(include_intercept=False)
    select_k = 2

    # Get filtered frame
    filtered_frame = display.frame(include_intercept=False, select_k=select_k)

    # For each estimator-label combination
    for estimator_name in filtered_frame["estimator"].unique():
        for label in filtered_frame["label"].unique():
            group_filtered = filtered_frame[
                (filtered_frame["estimator"] == estimator_name)
                & (filtered_frame["label"] == label)
            ]

            # Should have select_k features × n_splits rows
            n_splits = group_filtered["split"].nunique()
            assert group_filtered["feature"].nunique() == select_k, (
                f"Expected {select_k} unique features for {estimator_name}, "
                f"label {label}"
            )
            assert len(group_filtered) == select_k * n_splits, (
                f"Expected {select_k * n_splits} rows for {estimator_name}, "
                f"label {label}"
            )

            # Verify based on mean across splits for this group
            group_full = full_frame[
                (full_frame["estimator"] == estimator_name)
                & (full_frame["label"] == label)
            ]
            mean_abs_coefs = group_full.groupby("feature")["coefficients"].apply(
                lambda x: x.abs().mean()
            )
            expected_features = set(mean_abs_coefs.nlargest(select_k).index.tolist())
            actual_features = set(group_filtered["feature"].unique())
            assert actual_features == expected_features, (
                f"For {estimator_name}, label {label}, expected {expected_features}, "
                f"got {actual_features}"
            )


def test_sorting_order_descending(regression_data):
    """Test that sorting_order='descending' sorts per estimator by
    mean across splits."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="descending")

    # For each estimator, verify sorting by mean
    for estimator_name in sorted_frame["estimator"].unique():
        est_frame = sorted_frame[sorted_frame["estimator"] == estimator_name]
        mean_abs_coefs = est_frame.groupby("feature", sort=False)["coefficients"].apply(
            lambda x: x.abs().mean()
        )
        assert np.all(mean_abs_coefs.values[:-1] >= mean_abs_coefs.values[1:]), (
            f"Features not sorted in descending order for {estimator_name}"
        )


def test_sorting_order_ascending(regression_data):
    """Test that sorting_order='ascending' sorts per estimator by
    mean across splits."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    sorted_frame = display.frame(include_intercept=False, sorting_order="ascending")

    # For each estimator, verify sorting by mean
    for estimator_name in sorted_frame["estimator"].unique():
        est_frame = sorted_frame[sorted_frame["estimator"] == estimator_name]
        mean_abs_coefs = est_frame.groupby("feature", sort=False)["coefficients"].apply(
            lambda x: x.abs().mean()
        )
        assert np.all(mean_abs_coefs.values[:-1] <= mean_abs_coefs.values[1:]), (
            f"Features not sorted in ascending order for {estimator_name}"
        )


def test_select_k_with_sorting_descending(regression_data):
    """Test that select_k and sorting_order work together correctly."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    select_k = 3
    filtered_sorted_frame = display.frame(
        include_intercept=False, select_k=select_k, sorting_order="descending"
    )

    # For each estimator, verify both selection and sorting
    for estimator_name in filtered_sorted_frame["estimator"].unique():
        est_frame = filtered_sorted_frame[
            filtered_sorted_frame["estimator"] == estimator_name
        ]

        # Should have exactly select_k features × n_splits rows
        n_splits = est_frame["split"].nunique()
        assert est_frame["feature"].nunique() == select_k
        assert len(est_frame) == select_k * n_splits

        # Should be sorted in descending order by mean absolute coefficient
        mean_abs_coefs = est_frame.groupby("feature", sort=False)["coefficients"].apply(
            lambda x: x.abs().mean()
        )
        assert np.all(mean_abs_coefs.values[:-1] >= mean_abs_coefs.values[1:])


def test_select_k_negative_with_sorting_ascending(regression_data):
    """Test that negative select_k with ascending sort works correctly."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    bottom_k = 3
    filtered_sorted_frame = display.frame(
        include_intercept=False, select_k=-bottom_k, sorting_order="ascending"
    )

    # For each estimator, verify both selection and sorting
    for estimator_name in filtered_sorted_frame["estimator"].unique():
        est_frame = filtered_sorted_frame[
            filtered_sorted_frame["estimator"] == estimator_name
        ]

        # Should have exactly bottom_k features × n_splits rows
        n_splits = est_frame["split"].nunique()
        assert est_frame["feature"].nunique() == bottom_k
        assert len(est_frame) == bottom_k * n_splits

        # Should be sorted in ascending order by mean absolute coefficient
        mean_abs_coefs = est_frame.groupby("feature", sort=False)["coefficients"].apply(
            lambda x: x.abs().mean()
        )
        assert np.all(mean_abs_coefs.values[:-1] <= mean_abs_coefs.values[1:])


def test_plot_with_select_k(regression_data):
    """Test that plot method correctly uses select_k parameter."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    select_k = 3

    # Should not raise an error
    display.plot(select_k=select_k, include_intercept=False)

    # Verify the plot was created
    assert display.ax_ is not None
    assert display.figure_ is not None


def test_plot_with_sorting_order(regression_data):
    """Test that plot method correctly uses sorting_order parameter."""
    X, y = regression_data
    report_1 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report_2 = CrossValidationReport(Ridge(), X, y, splitter=2)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    display = report.feature_importance.coefficients()

    # Should not raise an error
    display.plot(sorting_order="ascending", include_intercept=False)

    # Verify the plot was created
    assert display.ax_ is not None
    assert display.figure_ is not None

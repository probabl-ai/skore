from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from skore import ComparisonReport, CrossValidationReport, EstimatorReport


def test_estimator(logistic_binary_classification_with_train_test):
    """Test that select_k works with EstimatorReport."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    coefficients = report.feature_importance.coefficients().frame(select_k=3)

    assert set(coefficients["feature"]) == {"Feature #10", "Feature #1", "Feature #15"}


def test_comparison_cross_validation(logistic_binary_classification_data):
    """Test that select_k works with ComparisonReports of CrossValidationReports."""
    estimator, X, y = logistic_binary_classification_data
    report_1 = CrossValidationReport(estimator, X, y)
    report_2 = CrossValidationReport(estimator, X, y)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    coefficients = report.feature_importance.coefficients().frame(select_k=3)

    assert set(coefficients["feature"]) == {"Feature #10", "Feature #1", "Feature #15"}


def test_positive(regression_train_test_split):
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


def test_negative(regression_train_test_split):
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


def test_multiclass(regression_train_test_split):
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


def test_plot(regression_train_test_split):
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

    display.plot(select_k=select_k, include_intercept=False)

    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")


def test_plot_different_features(logistic_binary_classification_with_train_test):
    """
    Test that select_k works correctly when the estimators have different features.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    report_2 = EstimatorReport(
        Pipeline([("poly", PolynomialFeatures()), ("predictor", clone(estimator))]),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.feature_importance.coefficients()
    display.plot(select_k=3)

    labels = [
        [tick_label.get_text() for tick_label in ax.get_yaxis().get_ticklabels()]
        for ax in display.ax_
    ]
    assert labels == [
        ["Feature #10", "Feature #1", "Feature #15"],
        ["Intercept", "x10", "x1"],
    ]

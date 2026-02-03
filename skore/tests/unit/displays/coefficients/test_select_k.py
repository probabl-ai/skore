from sklearn.linear_model import LogisticRegression

from skore import ComparisonReport, CrossValidationReport, EstimatorReport


def test_estimator(logistic_binary_classification_with_train_test):
    """`select_k` works with EstimatorReports."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    frame = report.inspection.coefficients().frame(select_k=3)

    assert set(frame["feature"]) == {"Feature #2", "Feature #3", "Feature #1"}


def test_cross_validation(logistic_binary_classification_data):
    """`select_k` works with CrossValidationReports."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X, y, splitter=2)

    frame = report.inspection.coefficients().frame(select_k=3)

    assert [list(group["feature"]) for _, group in frame.groupby("split")] == [
        ["Feature #3", "Feature #2", "Feature #1"],
        ["Feature #3", "Feature #2", "Feature #1"],
    ]


def test_comparison_cross_validation(logistic_binary_classification_data):
    """`select_k` works with ComparisonReports of CrossValidationReports."""
    estimator, X, y = logistic_binary_classification_data
    report_1 = CrossValidationReport(estimator, X, y)
    report_2 = CrossValidationReport(estimator, X, y)
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    coefficients = report.inspection.coefficients().frame(select_k=3)

    assert set(coefficients["feature"]) == {"Feature #2", "Feature #3", "Feature #1"}


def test_zero(comparison_report):
    """If `select_k` is zero then the output is an empty dataframe."""

    frame = comparison_report.inspection.coefficients().frame(select_k=0)
    assert frame.empty


def test_negative(comparison_report):
    """If `select_k` is negative then the features are the bottom `-select_k`."""
    frame = comparison_report.inspection.coefficients().frame(select_k=-3)
    assert set(frame["feature"]) == {"Feature #0", "Intercept", "Feature #1"}


def test_multiclass(multiclass_classification_train_test_split):
    """`select_k` works per estimator and per class in multiclass comparison."""
    X_train, X_test, y_train, y_test = multiclass_classification_train_test_split
    report_1 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report_2 = EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})
    frame = report.inspection.coefficients().frame(select_k=2)

    assert {
        (report, int(label)): list(group["feature"])
        for (report, label), group in frame.groupby(["estimator", "label"])
    } == {
        ("report_1", 0): ["Feature #3", "Feature #0"],
        ("report_1", 1): ["Feature #0", "Feature #3"],
        ("report_1", 2): ["Feature #3", "Feature #1"],
        ("report_2", 0): ["Feature #3", "Feature #0"],
        ("report_2", 1): ["Feature #0", "Feature #3"],
        ("report_2", 2): ["Feature #3", "Feature #1"],
    }


def test_multi_output_regression(linear_regression_multioutput_data):
    """`select_k` works per output in multi-output regression."""
    estimator, X, y = linear_regression_multioutput_data
    report = EstimatorReport(estimator, X_train=X, X_test=X, y_train=y, y_test=y)

    frame = report.inspection.coefficients().frame(select_k=2)

    assert [list(group["feature"]) for _, group in frame.groupby("output")] == [
        ["Feature #1", "Feature #0"],
        ["Feature #0", "Feature #3"],
    ]


def test_plot(comparison_report):
    """`select_k` works for plotting."""
    display = comparison_report.inspection.coefficients()

    display.plot(select_k=3)

    labels = [
        tick_label.get_text() for tick_label in display.ax_.get_yaxis().get_ticklabels()
    ]
    assert labels == ["Feature #3", "Feature #2", "Feature #1"]


def test_plot_different_features(comparison_report_different_features):
    """`select_k` works for plotting when the estimators have different features."""
    report = comparison_report_different_features

    display = report.inspection.coefficients()
    display.plot(select_k=3)

    labels = [
        [tick_label.get_text() for tick_label in ax.get_yaxis().get_ticklabels()]
        for ax in display.ax_
    ]
    assert labels == [
        ["Feature #3", "Feature #2", "Feature #1"],
        ["Feature #1", "Feature #0", "Intercept"],
    ]

import pytest
from sklearn.base import clone

from skore import ComparisonReport, EstimatorReport


@pytest.mark.parametrize("sorting_order", ["descending", "ascending"])
def test(comparison_report, sorting_order):
    """`sorting_order` sorts per estimator."""
    frame = comparison_report.feature_importance.coefficients().frame(
        sorting_order=sorting_order
    )

    for estimator in set(frame["estimator"]):
        coefs = list(frame.query(f"estimator == '{estimator}'")["coefficients"])
        assert coefs == sorted(coefs, key=abs, reverse=(sorting_order == "descending"))


def test_plot(comparison_report):
    """`sorting_order` works for plotting."""
    display = comparison_report.feature_importance.coefficients()

    display.plot(sorting_order="ascending")

    assert display.ax_ is not None
    assert display.figure_ is not None


@pytest.mark.parametrize("sorting_order", ["descending", "ascending"])
def test_plot_different_features(
    logistic_binary_classification_with_train_test, sorting_order
):
    """
    `sorting_order` works for plotting when the estimators have different features.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    # Limit the number of features to the first 5
    report_2 = EstimatorReport(
        clone(estimator),
        X_train=X_train[:, :4],
        X_test=X_test[:, :4],
        y_train=y_train,
        y_test=y_test,
    )
    report = ComparisonReport(reports={"report_1": report_1, "report_2": report_2})

    display = report.feature_importance.coefficients()
    display.plot(sorting_order=sorting_order, select_k=25)

    bar_widths = [[bar.get_width() for bar in ax.patches] for ax in display.ax_]
    assert len(bar_widths[0]) == 21
    assert len(bar_widths[1]) == 5
    for coefs in bar_widths:
        assert coefs == sorted(coefs, key=abs, reverse=(sorting_order == "descending"))

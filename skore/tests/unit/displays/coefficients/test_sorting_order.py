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
def test_plot_different_features(comparison_report_different_features, sorting_order):
    """
    `sorting_order` works for plotting when the estimators have different features.
    """
    report = comparison_report_different_features

    display = report.feature_importance.coefficients()
    display.plot(sorting_order=sorting_order, select_k=25)

    bar_widths = [[bar.get_width() for bar in ax.patches] for ax in display.ax_]
    assert len(bar_widths[0]) == 21
    assert len(bar_widths[1]) == 5 + 1  # 5 features + intercept
    for coefs in bar_widths:
        assert coefs == sorted(coefs, key=abs, reverse=(sorting_order == "descending"))

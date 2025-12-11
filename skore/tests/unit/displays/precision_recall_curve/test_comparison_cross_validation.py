import matplotlib.colors as mcolors
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from skore import ComparisonReport, CrossValidationReport
from skore._sklearn._plot.metrics.precision_recall_curve import (
    PrecisionRecallCurveDisplay,
)
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import (
    check_precision_recall_curve_display_data as check_display_data,
)


def test_binary_classification(
    pyplot, comparison_cross_validation_reports_binary_classification
):
    """
    Check the behaviour of `precision_recall` when ML task is "binary-classification".
    """
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    pos_label = 1
    n_reports = len(report.reports_)
    n_splits = len(next(iter(report.reports_.values())).estimator_reports_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * n_splits

    assert isinstance(display.ax_, np.ndarray)
    ax = display.ax_[0]
    check_legend_position(ax, loc="upper center", position="inside")
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]

    for estimator_name in report.reports_:
        average_precision = display.average_precision.query(
            f"label == {pos_label} & estimator_name == '{estimator_name}'"
        )["average_precision"]
        assert (
            f"{estimator_name} (AP={average_precision.mean():.2f}"
            f"±{average_precision.std():.2f})" in legend_texts
        )

    assert len(legend_texts) == n_reports
    assert ax.get_xlabel() == "recall"
    assert ax.get_ylabel() == "precision"
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert (
        display.figure_.get_suptitle()
        == f"Precision-Recall Curve\nPositive label: {pos_label}\nData source: Test set"
    )


def test_multiclass_classification(
    pyplot, comparison_cross_validation_reports_multiclass_classification
):
    """
    Check the behaviour of `precision_recall` when ML task is
    "multiclass-classification" and `pos_label` is None.
    """
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    labels = display.precision_recall["label"].cat.categories
    n_reports = len(report.reports_)
    n_splits = len(next(iter(report.reports_.values())).estimator_reports_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(labels) * n_reports * n_splits

    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(labels)

    for label, ax in zip(labels, display.ax_, strict=False):
        check_legend_position(ax, loc="upper center", position="inside")
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        for estimator_name in report.reports_:
            average_precision = display.average_precision.query(
                f"label == {label} & estimator_name == '{estimator_name}'"
            )["average_precision"]
            assert (
                f"{estimator_name} (AP={average_precision.mean():.2f}"
                f"±{average_precision.std():.2f})" in legend_texts
            )

        assert len(legend_texts) == n_reports
        assert ax.get_xlabel() == "recall"
        assert ax.get_ylabel() in ("precision", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert (
        display.figure_.get_suptitle()
        == "Precision-Recall Curve\nData source: Test set"
    )


def test_binary_classification_wrong_kwargs(
    pyplot, comparison_cross_validation_reports_binary_classification
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument."""
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.precision_recall()
    with pytest.raises(ValueError, match="subplot_by"):
        display.plot(subplot_by="invalid")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        display.plot(non_existent_kwarg="value")


def test_binary_classification_kwargs(
    pyplot, comparison_cross_validation_reports_binary_classification
):
    """Check that we can pass keyword arguments to the PR curve plot."""
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.precision_recall()

    n_reports = len(report.reports_)
    n_splits = len(next(iter(report.reports_.values())).estimator_reports_)

    display.plot()
    default_linewidth = display.lines_[0].get_linewidth()

    display.plot(relplot_kwargs={"linewidth": 2})
    assert len(display.lines_) == n_reports * n_splits
    for line in display.lines_:
        assert line.get_linewidth() == 2
        assert line.get_linewidth() != default_linewidth

    display.plot(relplot_kwargs={"alpha": 0.6})
    assert len(display.lines_) == n_reports * n_splits
    for line in display.lines_:
        assert line.get_alpha() == 0.6


def test_multiclass_classification_wrong_kwargs(
    pyplot, comparison_cross_validation_reports_multiclass_classification
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument."""
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.precision_recall()
    with pytest.raises(ValueError, match="subplot_by"):
        display.plot(subplot_by="invalid")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        display.plot(non_existent_kwarg="value")


def test_multiclass_classification_kwargs(
    pyplot, comparison_cross_validation_reports_multiclass_classification
):
    """Check that we can pass keyword arguments to the PR curve plot for
    multiclass classification."""
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.precision_recall()

    labels = display.precision_recall["label"].cat.categories
    n_reports = len(report.reports_)
    n_splits = len(next(iter(report.reports_.values())).estimator_reports_)

    display.plot(relplot_kwargs={"alpha": 0.5})
    assert len(display.lines_) == len(labels) * n_reports * n_splits
    for line in display.lines_:
        assert line.get_alpha() == 0.5

    display.plot(relplot_kwargs={"palette": ["red", "blue"]})
    assert len(display.lines_) == len(labels) * n_reports * n_splits

    expected_colors = ["red", "blue"]
    first_subplot_lines = display.ax_[0].get_lines()
    unique_colors = list(
        dict.fromkeys(line.get_color() for line in first_subplot_lines)
    )
    for expected_color, actual_color in zip(
        expected_colors, unique_colors, strict=True
    ):
        expected_rgb = mcolors.to_rgb(expected_color)
        actual_rgb = mcolors.to_rgb(actual_color)
        assert_allclose(expected_rgb, actual_rgb, atol=0.01)

    display.plot(despine=False)
    for ax in display.ax_:
        assert ax is not None


def test_binary_classification_constructor(forest_binary_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv + 1)
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()

    index_columns = ["estimator_name", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df.query("estimator_name == 'estimator_1'")[
            "split"
        ].unique().tolist() == list(range(cv))
        assert df.query("estimator_name == 'estimator_2'")[
            "split"
        ].unique().tolist() == list(range(cv + 1))
        assert df["estimator_name"].unique().tolist() == list(report.reports_.keys())
        assert df["label"].unique() == 1

    assert len(display.average_precision) == cv + (cv + 1)


def test_multiclass_classification_constructor(forest_multiclass_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = forest_multiclass_classification_data, 3
    report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv + 1)
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()

    index_columns = ["estimator_name", "split", "label"]
    classes = np.unique(y)
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df.query("estimator_name == 'estimator_1'")[
            "split"
        ].unique().tolist() == list(range(cv))
        assert df.query("estimator_name == 'estimator_2'")[
            "split"
        ].unique().tolist() == list(range(cv + 1))
        assert df["estimator_name"].unique().tolist() == list(report.reports_.keys())
        np.testing.assert_array_equal(df["label"].unique(), classes)

    assert len(display.average_precision) == len(classes) * cv + len(classes) * (cv + 1)


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_binary_classification(
    comparison_cross_validation_reports_binary_classification, with_average_precision
):
    """Test the frame method with binary classification comparison cross-validation
    data."""
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.precision_recall()

    df = display.frame(with_average_precision=with_average_precision)
    expected_index = ["estimator_name", "split"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator_name"].nunique() == len(report.reports_)

    if with_average_precision:
        for (_, _), group in df.groupby(["estimator_name", "split"], observed=True):
            assert group["average_precision"].nunique() == 1


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_multiclass_classification(
    comparison_cross_validation_reports_multiclass_classification,
    with_average_precision,
):
    """Test the frame method with multiclass classification comparison cross-validation
    data."""
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.precision_recall()

    df = display.frame(with_average_precision=with_average_precision)
    expected_index = ["estimator_name", "split", "label"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator_name"].nunique() == len(report.reports_)

    if with_average_precision:
        for (_, _, _), group in df.groupby(
            ["estimator_name", "split", "label"], observed=True
        ):
            assert group["average_precision"].nunique() == 1


def test_multiclass_str_labels_precision_recall_plot(pyplot):
    """Regression test for issue #2183 with multiclass comparison reports.

    Using string labels backed by numpy.str_ should not break
    `precision_recall().plot()` for a multiclass ComparisonReport.
    """
    iris = load_iris(as_frame=True)
    X, y_int = iris.data, iris.target
    y = iris.target_names[y_int]

    report_1 = CrossValidationReport(LogisticRegression(max_iter=200), X=X, y=y)
    report_2 = CrossValidationReport(LogisticRegression(max_iter=500), X=X, y=y)
    report = ComparisonReport([report_1, report_2])

    display = report.metrics.precision_recall()
    display.plot()

import re

import matplotlib as mpl
import numpy as np
import pytest
import seaborn as sns
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
    assert hasattr(display, "facet_")
    assert len(display.ax_) == n_reports

    for idx, estimator in enumerate(report.reports_):
        ax = display.ax_[idx]
        assert isinstance(ax, mpl.axes.Axes)
        check_legend_position(ax, loc="upper center", position="inside")
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"estimator == '{estimator}'")[
            "average_precision"
        ]
        assert (
            legend_texts[0] == f"AP={average_precision.mean():.2f}"
            f"±{average_precision.std():.2f}"
        )

        assert len(legend_texts) == 1
        assert ax.get_xlabel() == "recall"
        assert ax.get_ylabel() in ("precision", "")
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
    assert hasattr(display, "facet_")
    assert len(display.ax_) == n_reports

    for idx, estimator in enumerate(report.reports_):
        ax = display.ax_[idx]
        assert isinstance(ax, mpl.axes.Axes)
        check_legend_position(ax, loc="upper center", position="inside")
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        for label_idx, label in enumerate(labels):
            plot_data = display.frame(with_average_precision=True)
            average_precision = plot_data.query(
                f"label == {label} & estimator == '{estimator}'"
            )["average_precision"]
            assert (
                legend_texts[label_idx] == f"{label} (AP={average_precision.mean():.2f}"
                f"±{average_precision.std():.2f})"
            )

        assert len(legend_texts) == len(labels)
        assert ax.get_xlabel() == "recall"
        assert ax.get_ylabel() in ("precision", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert (
        display.figure_.get_suptitle()
        == "Precision-Recall Curve\nData source: Test set"
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "comparison_cross_validation_reports_binary_classification",
        "comparison_cross_validation_reports_multiclass_classification",
    ],
)
def test_wrong_kwargs(pyplot, fixture_name, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `relplot_kwargs` argument."""
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.precision_recall()
    err_msg = "Line2D.set() got an unexpected keyword argument 'invalid'"
    with pytest.raises(AttributeError, match=re.escape(err_msg)):
        display.set_style(relplot_kwargs={"invalid": "value"}).plot()


@pytest.mark.parametrize(
    "fixture_name",
    [
        "comparison_cross_validation_reports_binary_classification",
        "comparison_cross_validation_reports_multiclass_classification",
    ],
)
def test_relplot_kwargs(pyplot, fixture_name, request):
    """Check that we can pass keyword arguments to the PR curve plot."""
    report = request.getfixturevalue(fixture_name)
    multiclass = "multiclass" in fixture_name

    display = report.metrics.precision_recall()

    display.plot()
    assert hasattr(display, "facet_")

    if multiclass:
        display.set_style(relplot_kwargs={"palette": ["red", "blue", "green"]}).plot()
    else:
        display.set_style(relplot_kwargs={"color": "red"}).plot()
    assert hasattr(display, "facet_")


def test_binary_classification_constructor(forest_binary_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = forest_binary_classification_data, 3
    report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv + 1)
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()

    index_columns = ["estimator", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df.query("estimator == 'estimator_1'")[
            "split"
        ].unique().tolist() == list(range(cv))
        assert df.query("estimator == 'estimator_2'")[
            "split"
        ].unique().tolist() == list(range(cv + 1))
        assert df["estimator"].unique().tolist() == list(report.reports_.keys())
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

    index_columns = ["estimator", "split", "label"]
    classes = np.unique(y)
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df.query("estimator == 'estimator_1'")[
            "split"
        ].unique().tolist() == list(range(cv))
        assert df.query("estimator == 'estimator_2'")[
            "split"
        ].unique().tolist() == list(range(cv + 1))
        assert df["estimator"].unique().tolist() == list(report.reports_.keys())
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
    expected_index = ["estimator", "split"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == len(report.reports_)

    if with_average_precision:
        for (_, _), group in df.groupby(["estimator", "split"], observed=True):
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
    expected_index = ["estimator", "split", "label"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == len(report.reports_)

    if with_average_precision:
        for (_, _, _), group in df.groupby(
            ["estimator", "split", "label"], observed=True
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

    report_1 = CrossValidationReport(LogisticRegression(max_iter=500), X=X, y=y)
    report_2 = CrossValidationReport(LogisticRegression(max_iter=1000), X=X, y=y)
    report = ComparisonReport([report_1, report_2])

    display = report.metrics.precision_recall()
    display.plot()


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        (
            "comparison_cross_validation_reports_binary_classification",
            ["None", "auto", "estimator"],
        ),
        (
            "comparison_cross_validation_reports_multiclass_classification",
            ["auto", "estimator", "label"],
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.precision_recall()
    valid_values_str = ", ".join(valid_values)
    err_msg = f"subplot_by must be one of {valid_values_str}. Got 'invalid' instead."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "comparison_cross_validation_reports_binary_classification",
            [(None, 0), ("estimator", 2)],
        ),
        (
            "comparison_cross_validation_reports_multiclass_classification",
            [("label", 3), ("estimator", 2)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)
    display = report.metrics.precision_recall()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len

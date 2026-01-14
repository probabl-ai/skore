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
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * n_splits

    assert len(display.ax_) == n_reports

    expected_colors = sns.color_palette()[:1]
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
        for line in ax.get_lines():
            assert line.get_color() == expected_colors[0]

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
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * len(labels) * n_splits

    assert len(display.ax_) == n_reports

    expected_colors = sns.color_palette()[: len(labels)]
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
            lines_slice = ax.get_lines()[
                label_idx * n_splits : (label_idx + 1) * n_splits
            ]
            for line in lines_slice:
                assert line.get_color() == expected_colors[label_idx]

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
        display.plot(relplot_kwargs={"invalid": "value"})


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
    n_reports = len(report.reports_)
    n_splits = len(next(iter(report.reports_.values())).estimator_reports_)
    n_labels = (
        len(display.precision_recall["label"].cat.categories) if multiclass else 1
    )

    display.plot()
    default_colors = [line.get_color() for line in display.lines_]
    if multiclass:
        # With subplot_by="estimator", lines are organized per subplot:
        # For each subplot: [label0_color] * n_splits + [label1_color] * n_splits + ...
        palette_colors = sns.color_palette()[:n_labels]
        expected_default = sum([[c] * n_splits for c in palette_colors], []) * n_reports
    else:
        # Binary: each subplot has n_splits lines, all same color
        expected_default = [sns.color_palette()[0]] * n_splits * n_reports
    assert default_colors == expected_default

    if multiclass:
        # For multiclass, use palette since there's a hue variable
        palette_colors = ["red", "blue", "green"]
        display.plot(relplot_kwargs={"palette": palette_colors})
        assert len(display.lines_) == n_reports * n_splits * n_labels
        expected_colors = sum([[c] * n_splits for c in palette_colors], []) * n_reports
        for line, expected_color, default_color in zip(
            display.lines_, expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color

        palette_colors = ["magenta", "cyan", "yellow"]
        display.set_style(relplot_kwargs={"palette": palette_colors}, policy="update")
        display.plot()
        assert len(display.lines_) == n_reports * n_splits * n_labels
        expected_colors = sum([[c] * n_splits for c in palette_colors], []) * n_reports
        for line, expected_color, default_color in zip(
            display.lines_, expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color
    else:
        # For binary, use color since there's no hue variable
        display.plot(relplot_kwargs={"color": "red"})
        assert len(display.lines_) == n_reports * n_splits * n_labels
        expected_colors = ["red"] * n_splits * n_reports
        for line, expected_color, default_color in zip(
            display.lines_, expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color

        display.set_style(relplot_kwargs={"color": "green"}, policy="update")
        display.plot()
        assert len(display.lines_) == n_reports * n_splits * n_labels
        expected_colors = ["green"] * n_splits * n_reports
        for line, expected_color, default_color in zip(
            display.lines_, expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color


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

    report_1 = CrossValidationReport(LogisticRegression(), X=X, y=y)
    report_2 = CrossValidationReport(LogisticRegression(max_iter=500), X=X, y=y)
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

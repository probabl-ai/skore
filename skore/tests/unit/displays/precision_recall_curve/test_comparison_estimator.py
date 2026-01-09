import re

import matplotlib as mpl
import numpy as np
import pytest
import seaborn as sns
from sklearn.base import clone

from skore import ComparisonReport, EstimatorReport
from skore._sklearn._plot import PrecisionRecallCurveDisplay
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import (
    check_precision_recall_curve_display_data as check_display_data,
)


def test_binary_classification(pyplot, logistic_binary_classification_with_train_test):
    """Check the attributes and default plotting behaviour of the precision-recall curve
    plot with binary data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)
    n_reports = len(report.reports_)

    display.plot()
    assert len(display.ax_) == n_reports

    expected_colors = sns.color_palette()[:1]
    for idx, estimator in enumerate(report.reports_):
        ax = display.ax_[idx]
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        plot_data = display.frame(with_average_precision=True)
        average_precision = plot_data.query(f"estimator == '{estimator}'")[
            "average_precision"
        ].iloc[0]
        assert legend_texts[0] == f"AP={average_precision:.2f}"

        line = display.lines_[idx]
        assert isinstance(line, mpl.lines.Line2D)
        assert line.get_color() == expected_colors[0]

        assert len(legend_texts) == 1
        assert ax.get_xlabel() == "recall"
        assert ax.get_ylabel() in ("precision", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert (
        display.figure_.get_suptitle() == f"Precision-Recall Curve"
        f"\nPositive label: {display.pos_label}"
        f"\nData source: Test set"
    )


def test_multiclass_classification(
    pyplot, logistic_multiclass_classification_with_train_test
):
    """Check the attributes and default plotting behaviour of the precision-recall curve
    plot with multiclass data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    class_labels = next(iter(report.reports_.values())).estimator_.classes_
    n_reports = len(report.reports_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * n_reports
    expected_colors = sns.color_palette()[: len(class_labels)]
    assert len(display.ax_) == n_reports

    for idx, estimator in enumerate(report.reports_):
        ax = display.ax_[idx]
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        for class_label_idx, class_label in enumerate(class_labels):
            plot_data = display.frame(with_average_precision=True)
            average_precision = plot_data.query(
                f"label == {class_label} & estimator == '{estimator}'"
            )["average_precision"].iloc[0]
            assert (
                legend_texts[class_label_idx]
                == f"{class_label} (AP={average_precision:.2f})"
            )
            line = ax.get_lines()[class_label_idx]
            assert line.get_color() == expected_colors[class_label_idx]

        assert len(legend_texts) == len(class_labels)
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
        "logistic_binary_classification_with_train_test",
        "logistic_multiclass_classification_with_train_test",
    ],
)
def test_wrong_kwargs(pyplot, fixture_name, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `relplot_kwargs` argument."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)

    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    err_msg = "Line2D.set() got an unexpected keyword argument 'invalid'"
    with pytest.raises(AttributeError, match=re.escape(err_msg)):
        display.plot(relplot_kwargs={"invalid": "value"})


@pytest.mark.parametrize(
    "fixture_name",
    [
        "logistic_binary_classification_with_train_test",
        "logistic_multiclass_classification_with_train_test",
    ],
)
def test_relplot_kwargs(pyplot, fixture_name, request):
    """Check that we can pass keyword arguments to the precision-recall curve plot."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    multiclass = "multiclass" in fixture_name
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    n_reports = len(report.reports_)
    n_labels = (
        len(display.precision_recall["label"].cat.categories) if multiclass else 1
    )

    display.plot()
    default_colors = [line.get_color() for line in display.lines_]
    if multiclass:
        # With subplot_by="estimator", each subplot has n_labels lines
        # Colors cycle by label: [label0_color, label1_color, ...] per subplot
        palette_colors = sns.color_palette()[:n_labels]
        expected_default = palette_colors * n_reports
    else:
        # Binary: each subplot has 1 line, all same color
        expected_default = [sns.color_palette()[0]] * n_reports
    assert default_colors == expected_default

    if multiclass:
        # For multiclass, palette works since there's a hue variable
        display.plot(relplot_kwargs={"palette": ["red", "blue"]})
        assert len(display.lines_) == n_reports * n_labels
        # Colors: red for first label, blue for second, default for third if exists
        palette_colors = ["red", "blue"] + list(sns.color_palette()[2:n_labels])
        expected_colors = palette_colors[:n_labels] * n_reports
        for line, expected_color, default_color in zip(
            display.lines_, expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color

        display.set_style(
            relplot_kwargs={"palette": ["green", "yellow"]}, policy="update"
        )
        display.plot()
        assert len(display.lines_) == n_reports * n_labels
        palette_colors = ["green", "yellow"] + list(sns.color_palette()[2:n_labels])
        expected_colors = palette_colors[:n_labels] * n_reports
        for line, expected_color, default_color in zip(
            display.lines_, expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color
    else:
        # For binary, palette is ignored (no hue), so use color instead
        display.plot(relplot_kwargs={"color": "red"})
        assert len(display.lines_) == n_reports * n_labels
        expected_colors = ["red"] * n_reports
        for line, expected_color, default_color in zip(
            display.lines_, expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color

        display.set_style(relplot_kwargs={"color": "green"}, policy="update")
        display.plot()
        assert len(display.lines_) == n_reports * n_labels
        expected_colors = ["green"] * n_reports
        for line, expected_color, default_color in zip(
            display.lines_, expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_binary_classification(
    logistic_binary_classification_with_train_test, with_average_precision
):
    """Test the frame method with binary classification comparison data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    df = display.frame(with_average_precision=with_average_precision)

    expected_index = ["estimator"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == 2

    if with_average_precision:
        for (_), group in df.groupby(["estimator"], observed=True):
            assert group["average_precision"].nunique() == 1


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_multiclass_classification(
    logistic_multiclass_classification_with_train_test, with_average_precision
):
    """Test the frame method with multiclass classification comparison data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.precision_recall()
    df = display.frame(with_average_precision=with_average_precision)

    expected_index = ["estimator", "label"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == 2

    if with_average_precision:
        for (_, _), group in df.groupby(["estimator", "label"], observed=True):
            assert group["average_precision"].nunique() == 1


def test_legend(
    pyplot,
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check the rendering of the legend for with a `ComparisonReport`."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_[0], loc="upper center", position="inside")

    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_[0], loc="upper center", position="inside")


def test_binary_classification_constructor(
    logistic_binary_classification_with_train_test,
):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()

    index_columns = ["estimator", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique().tolist() == list(report.reports_.keys())
        assert df["split"].isnull().all()
        assert df["label"].unique() == 1

    assert len(display.average_precision) == 2


def test_multiclass_classification_constructor(
    logistic_multiclass_classification_with_train_test,
):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()

    index_columns = ["estimator", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique().tolist() == list(report.reports_.keys())
        assert df["split"].isnull().all()
        np.testing.assert_array_equal(df["label"].unique(), np.unique(y_train))

    assert len(display.average_precision) == len(np.unique(y_train)) * 2


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        (
            "logistic_binary_classification_with_train_test",
            ["None", "auto", "estimator"],
        ),
        (
            "logistic_multiclass_classification_with_train_test",
            ["auto", "estimator", "label"],
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()
    valid_values_str = ", ".join(valid_values)
    err_msg = f"subplot_by must be one of {valid_values_str}. Got 'invalid' instead."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "logistic_binary_classification_with_train_test",
            [(None, 0), ("estimator", 2)],
        ),
        (
            "logistic_multiclass_classification_with_train_test",
            [("label", 3), ("estimator", 2)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.precision_recall()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len

"""Test the `roc_curve` display method."""

import re

import matplotlib as mpl
import numpy as np
import pytest
import seaborn as sns

from skore import ComparisonReport, CrossValidationReport
from skore._sklearn._plot.metrics.roc_curve import RocCurveDisplay
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import check_roc_curve_display_data as check_display_data


def test_binary_classification(
    pyplot, comparison_cross_validation_reports_binary_classification
):
    """Check the behaviour of `roc_curve` when ML task is "binary-classification"."""
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)
    check_display_data(display)

    pos_label = 1
    n_reports = len(report.reports_)
    n_splits = len(next(iter(report.reports_.values())).estimator_reports_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * n_splits + n_reports

    assert len(display.ax_) == n_reports

    expected_colors = sns.color_palette()[:1]
    for idx, estimator in enumerate(report.reports_):
        ax = display.ax_[idx]
        assert isinstance(ax, mpl.axes.Axes)
        check_legend_position(ax, loc="upper center", position="inside")
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"estimator == '{estimator}'")["roc_auc"]
        assert legend_texts[0] == f"AUC={roc_auc.mean():.2f}±{roc_auc.std():.2f}"
        for line in ax.get_lines()[:-1]:
            assert line.get_color() == expected_colors[0]

        assert len(legend_texts) == 1
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() in ("True Positive Rate", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert (
        display.figure_.get_suptitle()
        == f"ROC Curve\nPositive label: {pos_label}\nData source: Test set"
    )


def test_multiclass_classification(
    pyplot, comparison_cross_validation_reports_multiclass_classification
):
    """Check the behaviour of `roc_curve` when ML task is "multiclass-classification"
    and `pos_label` is None."""
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)
    check_display_data(display)

    labels = display.roc_curve["label"].cat.categories
    n_reports = len(report.reports_)
    n_splits = len(next(iter(report.reports_.values())).estimator_reports_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * len(labels) * n_splits + n_reports

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
            plot_data = display.frame(with_roc_auc=True)
            roc_auc = plot_data.query(f"label == {label} & estimator == '{estimator}'")[
                "roc_auc"
            ]
            assert (
                legend_texts[label_idx] == f"{label} (AUC={roc_auc.mean():.2f}"
                f"±{roc_auc.std():.2f})"
            )
            lines_slice = ax.get_lines()[
                label_idx * n_splits : (label_idx + 1) * n_splits
            ]
            for line in lines_slice:
                assert line.get_color() == expected_colors[label_idx]

        assert len(legend_texts) == len(labels)
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() in ("True Positive Rate", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert display.figure_.get_suptitle() == "ROC Curve\nData source: Test set"


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
    display = report.metrics.roc()
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
    """Check that we can pass keyword arguments to the ROC curve plot."""
    report = request.getfixturevalue(fixture_name)
    multiclass = "multiclass" in fixture_name

    display = report.metrics.roc()
    n_reports = len(report.reports_)
    n_splits = len(next(iter(report.reports_.values())).estimator_reports_)
    n_labels = len(display.roc_curve["label"].cat.categories) if multiclass else 1

    display.plot()
    n_roc_lines = n_reports * n_splits * n_labels
    default_colors = [line.get_color() for line in display.lines_[:n_roc_lines]]
    if multiclass:
        palette_colors = sns.color_palette()[:n_labels]
        expected_default = sum([[c] * n_splits for c in palette_colors], []) * n_reports
    else:
        expected_default = [sns.color_palette()[0]] * n_splits * n_reports
    assert default_colors == expected_default

    if multiclass:
        palette_colors = ["red", "blue", "green"]
        display.set_style(relplot_kwargs={"palette": palette_colors}).plot()
        assert len(display.lines_) == n_roc_lines + n_reports
        expected_colors = sum([[c] * n_splits for c in palette_colors], []) * n_reports
        for line, expected_color, default_color in zip(
            display.lines_[:n_roc_lines], expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color

    else:
        display.set_style(relplot_kwargs={"color": "red"}).plot()
        assert len(display.lines_) == n_roc_lines + n_reports
        expected_colors = ["red"] * n_splits * n_reports
        for line, expected_color, default_color in zip(
            display.lines_[:n_roc_lines], expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color


def test_binary_classification_constructor(logistic_binary_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv + 1)
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()

    index_columns = ["estimator", "split", "label"]
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df.query("estimator == 'estimator_1'")[
            "split"
        ].unique().tolist() == list(range(cv))
        assert df.query("estimator == 'estimator_2'")[
            "split"
        ].unique().tolist() == list(range(cv + 1))
        assert df["estimator"].unique().tolist() == list(report.reports_.keys())
        assert df["label"].unique() == 1

    assert len(display.roc_auc) == cv + (cv + 1)


def test_multiclass_classification_constructor(logistic_multiclass_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv + 1)
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()

    index_columns = ["estimator", "split", "label"]
    classes = np.unique(y)
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df.query("estimator == 'estimator_1'")[
            "split"
        ].unique().tolist() == list(range(cv))
        assert df.query("estimator == 'estimator_2'")[
            "split"
        ].unique().tolist() == list(range(cv + 1))
        assert df["estimator"].unique().tolist() == list(report.reports_.keys())
        np.testing.assert_array_equal(df["label"].unique(), classes)

    assert len(display.roc_auc) == len(classes) * cv + len(classes) * (cv + 1)


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_binary_classification(
    comparison_cross_validation_reports_binary_classification, with_roc_auc
):
    """Test the frame method with binary classification comparison
    cross-validation data.
    """
    report = comparison_cross_validation_reports_binary_classification
    display = report.metrics.roc()
    df = display.frame(with_roc_auc=with_roc_auc)

    expected_index = ["estimator", "split"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == len(report.reports_)

    if with_roc_auc:
        for (_, _), group in df.groupby(["estimator", "split"], observed=True):
            assert group["roc_auc"].nunique() == 1


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_multiclass_classification(
    comparison_cross_validation_reports_multiclass_classification, with_roc_auc
):
    """Test the frame method with multiclass classification comparison
    cross-validation data.
    """
    report = comparison_cross_validation_reports_multiclass_classification
    display = report.metrics.roc()
    df = display.frame(with_roc_auc=with_roc_auc)

    expected_index = ["estimator", "split", "label"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == len(report.reports_)

    if with_roc_auc:
        for (_, _, _), group in df.groupby(
            ["estimator", "split", "label"], observed=True
        ):
            assert group["roc_auc"].nunique() == 1


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
    display = report.metrics.roc()
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
    display = report.metrics.roc()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len

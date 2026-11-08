import re

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import CrossValidationReport
from skore._displays import RocCurveDisplay
from skore._utils.testing import check_roc_curve_display_data


@pytest.fixture
def cross_validation_report_aggregate():
    """Report whose splits have ROC curves distinct enough to tell them apart.

    The default binary classification data is separable, which makes every AUC 1.00 and
    therefore makes the aggregated curve indistinguishable from the ones of the splits.
    """
    X, y = make_classification(
        n_samples=200, n_features=4, class_sep=0.5, random_state=42
    )
    return CrossValidationReport(LogisticRegression(), X, y, splitter=3)


def _nearest_index(thresholds: np.ndarray, target: float) -> int:
    """Index of the nearest threshold, matching the display's aggregation rule."""
    insertion = int(np.searchsorted(thresholds, target, side="left"))
    insertion = min(insertion, len(thresholds) - 1)
    if thresholds[insertion] == target or insertion == 0:
        return insertion
    previous = insertion - 1
    higher, lower = thresholds[insertion], thresholds[previous]
    if np.isfinite(higher) and np.isfinite(lower) and np.isfinite(target):
        if abs(higher - target) <= abs(lower - target):
            return insertion
        return previous
    # +inf is farther from a finite target than the last finite threshold is, and a
    # target at +inf falls on the largest threshold of a split that has no +inf.
    return insertion if np.isfinite(higher) else previous


def _lines_matching(ax, curve):
    """Get the lines of `ax` drawing the (fpr, tpr) points of `curve`."""
    return [
        line
        for line in ax.get_lines()
        if len(line.get_xdata()) == len(curve)
        and np.allclose(line.get_xdata(), curve["fpr"])
        and np.allclose(line.get_ydata(), curve["tpr"])
    ]


def test_legend_binary_classification(
    cross_validation_reports_binary_classification,
    cross_validation_reports_binary_classification_figure_axes,
):
    """Check the legend of the ROC curve plot with binary data."""
    report = cross_validation_reports_binary_classification[0]
    display = report.metrics.roc()
    _, ax = cross_validation_reports_binary_classification_figure_axes
    legend = ax[0].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    labels = display.roc_curve["label"].cat.categories
    for label_idx, label in enumerate(labels):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {label}")["roc_auc"]
        assert (
            legend_texts[label_idx] == f"{label} (AUC={roc_auc.mean():.2f}"
            f"±{roc_auc.std():.2f})"
        )
    assert len(legend_texts) == len(labels) + 1
    assert legend_texts[-1] == "Chance level (AUC = 0.5)"


def test_no_aggregate(cross_validation_reports_binary_classification):
    """Check that without `aggregate`, only the curves of the splits are computed."""
    report = cross_validation_reports_binary_classification[0]
    display = report.metrics.roc()

    check_roc_curve_display_data(display)
    assert display.roc_curve["split"].notna().all()


def test_aggregate_threshold_frame(cross_validation_report_aggregate):
    """Check the curve that `aggregate="threshold"` adds to the curves of the splits."""
    display = cross_validation_report_aggregate.metrics.roc(aggregate="threshold")

    check_roc_curve_display_data(display, aggregate=True)
    curve, roc_auc = display.roc_curve, display.roc_auc
    assert curve[curve["aggregate"].isna()]["split"].notna().all()
    assert curve[curve["aggregate"].notna()]["split"].isna().all()

    for label in curve["label"].cat.categories:
        aggregated = curve[curve["aggregate"].notna() & (curve["label"] == label)]
        splits = [
            split_curve.sort_values("threshold", ascending=False)
            for _, split_curve in curve[
                curve["aggregate"].isna() & (curve["label"] == label)
            ].groupby("split", observed=True)
        ]

        threshold_sets = [
            {
                threshold
                for threshold in split["threshold"].tolist()
                if np.isfinite(threshold)
            }
            for split in splits
        ]
        shared = set.intersection(*threshold_sets)
        grid = sorted(shared or set.union(*threshold_sets), reverse=True)
        assert_allclose(aggregated["threshold"], grid)

        # each split contributes the point at its nearest threshold; an equal
        # distance keeps the higher one
        splits_ascending = [split.sort_values("threshold") for split in splits]
        points = (point for _, point in aggregated.iterrows())
        for threshold, point in zip(grid, points, strict=True):
            fprs, tprs = [], []
            for split in splits_ascending:
                index = _nearest_index(split["threshold"].to_numpy(), threshold)
                fprs.append(split["fpr"].iloc[index])
                tprs.append(split["tpr"].iloc[index])
            assert point["fpr"] == pytest.approx(np.mean(fprs))
            assert point["tpr"] == pytest.approx(np.mean(tprs))

        # the reported AUC is the mean of the per-split AUCs
        split_auc = roc_auc[roc_auc["aggregate"].isna() & (roc_auc["label"] == label)][
            "roc_auc"
        ]
        aggregated_auc = roc_auc[
            roc_auc["aggregate"].notna() & (roc_auc["label"] == label)
        ]["roc_auc"]
        assert aggregated_auc.item() == pytest.approx(split_auc.mean())


def test_aggregate_threshold_nearest_on_the_union():
    """With no shared threshold, each split contributes its nearest point.

    The operating points are not rates. At threshold 1 the second split is equally
    far from 0 and 2, and the higher threshold wins, so it contributes 3 rather than 1.
    """
    splits = {
        0: {"threshold": [0, 4], "value": [1, 5]},
        1: {"threshold": [2, 5], "value": [3, 7]},
    }
    curve_rows = [
        {"split": split, "threshold": threshold, "fpr": value, "tpr": value}
        for split, points in splits.items()
        for threshold, value in zip(points["threshold"], points["value"], strict=True)
    ]
    curve = pd.DataFrame(curve_rows)
    curve["estimator"] = "estimator"
    curve["data_source"] = "test"
    curve["label"] = 1
    for column in ("estimator", "data_source", "split", "label"):
        curve[column] = curve[column].astype("category")

    auc_frame = curve.drop_duplicates("split").drop(columns=["threshold", "fpr", "tpr"])
    auc_frame["roc_auc"] = [0.2, 0.4]

    display = RocCurveDisplay(
        roc_curve=curve,
        roc_auc=auc_frame,
        report_pos_label=1,
        data_source="test",
        ml_task="binary-classification",
        report_type="cross-validation",
    )._aggregate_over_splits()

    aggregated = display.roc_curve[display.roc_curve["aggregate"].notna()]
    # union is 0, 2, 4, 5, stored in decreasing order
    # t=5 -> split 0 at 4 (5), split 1 at 5 (7)
    # t=4 -> split 0 at 4 (5), split 1 at 5 (7); 5 is closer than 2
    # t=2 -> split 0 at 0 or 4, equal distance, higher wins (5); split 1 at 2 (3)
    # t=0 -> split 0 at 0 (1), split 1 at 2 (3)
    assert_allclose(aggregated["threshold"], [5, 4, 2, 0])
    assert_allclose(aggregated["fpr"], [6, 6, 4, 2])
    assert_allclose(aggregated["tpr"], [6, 6, 4, 2])
    aggregated_auc = display.roc_auc[display.roc_auc["aggregate"].notna()]["roc_auc"]
    assert aggregated_auc.item() == pytest.approx(0.3)


def test_aggregate_threshold_plot(cross_validation_report_aggregate):
    """Check that the aggregated curve is drawn as computed, over the splits."""
    display = cross_validation_report_aggregate.metrics.roc(aggregate="threshold")
    curve = display.roc_curve
    ax = display.plot(label=1).axes[0]

    aggregated = curve[curve["aggregate"].notna() & (curve["label"] == 1)]
    (line,) = _lines_matching(ax, aggregated)
    # ROC curves are step functions, and seaborn must not aggregate the points it is
    # given, which would flatten every vertical rise of the curve and add an error band
    assert line.get_drawstyle() == "steps-post"
    assert len(ax.collections) == 0

    for _, split_curve in curve[
        curve["aggregate"].isna() & (curve["label"] == 1)
    ].groupby("split", observed=True):
        (split_line,) = _lines_matching(ax, split_curve)
        assert split_line.get_alpha() == 0.4


@pytest.mark.parametrize(
    "label, expected_entry",
    [
        (1, "AUC={auc} (threshold-averaged)"),
        (None, "1 (AUC={auc}, threshold-averaged)"),
    ],
    ids=["single-label", "one-vs-rest"],
)
def test_aggregate_threshold_legend(
    cross_validation_report_aggregate, label, expected_entry
):
    """Check that the legend reports the mean and std of the per-split AUCs."""
    display = cross_validation_report_aggregate.metrics.roc(aggregate="threshold")
    ax = display.plot(label=label).axes[0]

    plot_data = display.frame(with_roc_auc=True, label=label)
    split_auc = plot_data[plot_data["aggregate"].isna()]
    if "label" in split_auc.columns:
        split_auc = split_auc[split_auc["label"] == 1]
    split_auc = split_auc["roc_auc"]
    statistic = f"{split_auc.mean():.2f}±{split_auc.std():.2f}"

    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert expected_entry.format(auc=statistic) in legend_texts
    assert legend_texts[-1] == "Chance level (AUC = 0.5)"


def test_aggregate_threshold_style(cross_validation_report_aggregate):
    """Check that the aggregated curve is styled like the splits it is drawn over."""
    display = cross_validation_report_aggregate.metrics.roc(aggregate="threshold")
    display.set_style(relplot_kwargs={"palette": ["red", "green"], "linewidth": 4})
    curve = display.roc_curve
    ax = display.plot(label=None).axes[0]

    labels = curve["label"].cat.categories
    for label, color in zip(labels, ["red", "green"], strict=True):
        is_label = curve["label"] == label
        (line,) = _lines_matching(ax, curve[curve["aggregate"].notna() & is_label])
        assert line.get_color() == color
        assert line.get_linewidth() == 4
        # the aggregated curve stands out by being opaque, not by being thicker
        assert line.get_alpha() is None


def test_aggregate_threshold_data_source_both(cross_validation_report_aggregate):
    """Check that the aggregated curve of a data source is dashed as its splits."""
    display = cross_validation_report_aggregate.metrics.roc(
        data_source="both", aggregate="threshold"
    )
    curve = display.roc_curve
    ax = display.plot(label=1).axes[0]

    for data_source in ("train", "test"):
        is_group = (curve["data_source"] == data_source) & (curve["label"] == 1)
        (line,) = _lines_matching(ax, curve[curve["aggregate"].notna() & is_group])
        split_lines = [
            _lines_matching(ax, split_curve)[0]
            for _, split_curve in curve[curve["aggregate"].isna() & is_group].groupby(
                "split", observed=True
            )
        ]
        assert {line.get_linestyle()} == {
            split_line.get_linestyle() for split_line in split_lines
        }


def test_aggregate_hides_splits(cross_validation_report_aggregate):
    """Check that the splits can be hidden, leaving only the aggregated curve."""
    display = cross_validation_report_aggregate.metrics.roc(aggregate="threshold")
    display.set_style(relplot_kwargs={"alpha": 0.0})
    curve = display.roc_curve
    ax = display.plot(label=1).axes[0]

    is_label_1 = curve["label"] == 1
    (line,) = _lines_matching(ax, curve[curve["aggregate"].notna() & is_label_1])
    assert line.get_alpha() is None
    for _, split_curve in curve[
        curve["aggregate"].isna() & (curve["label"] == 1)
    ].groupby("split", observed=True):
        (split_line,) = _lines_matching(ax, split_curve)
        assert split_line.get_alpha() == 0.0


def test_aggregate_invalid_value(cross_validation_report_aggregate):
    """Check the error message when passing an unknown `aggregate`."""
    err_msg = 'aggregate must be "threshold" or None.'
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        cross_validation_report_aggregate.metrics.roc(aggregate="invalid")


def test_aggregate_multiclass(cross_validation_reports_multiclass_classification):
    """Check the error message when aggregating in multiclass classification."""
    report = cross_validation_reports_multiclass_classification[0]
    err_msg = "Averaging is not implemented for multiclass classification"
    with pytest.raises(ValueError, match=err_msg):
        report.metrics.roc(aggregate="threshold")


def test_legend_multiclass_classification(
    cross_validation_reports_multiclass_classification,
    cross_validation_reports_multiclass_classification_figure_axes,
):
    """Check the legend of the ROC curve plot with multiclass data."""
    report = cross_validation_reports_multiclass_classification[0]
    display = report.metrics.roc()
    _, ax = cross_validation_reports_multiclass_classification_figure_axes
    labels = display.roc_curve["label"].cat.categories
    legend = ax[0].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    for label_idx, label in enumerate(labels):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {label}")["roc_auc"]
        assert (
            legend_texts[label_idx] == f"{label} (AUC={roc_auc.mean():.2f}"
            f"±{roc_auc.std():.2f})"
        )
    assert len(legend_texts) == len(labels) + 1
    assert legend_texts[-1] == "Chance level (AUC = 0.5)"


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        ("cross_validation_reports_binary_classification", ["None", "auto", "label"]),
        (
            "cross_validation_reports_multiclass_classification",
            ["None", "auto", "label"],
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    report = request.getfixturevalue(fixture_name)[0]

    display = report.metrics.roc()
    err_msg = (
        f"subplot_by must be one of {', '.join(valid_values)}. Got 'invalid' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        ("cross_validation_reports_binary_classification", [(None, 0), ("label", 2)]),
        (
            "cross_validation_reports_multiclass_classification",
            [("label", 3)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    report = request.getfixturevalue(fixture_name)[0]
    display = report.metrics.roc()
    for subplot_by, expected_len in subplot_by_tuples:
        fig = display.plot(subplot_by=subplot_by)
        axes = fig.axes
        if subplot_by is None:
            assert len(axes) == 1
            assert isinstance(axes[0], mpl.axes.Axes)
        else:
            assert len(axes) == expected_len


@pytest.mark.parametrize(
    "report_name",
    [
        "cross_validation_reports_binary_classification",
        "cross_validation_reports_multiclass_classification",
    ],
)
def test_data_source_both(report_name, request):
    """Check that `roc(data_source='both')` includes train and test data."""
    report = request.getfixturevalue(report_name)[0]
    display = report.metrics.roc(data_source="both")

    assert display.data_source == "both"
    frame = display.frame()
    assert "data_source" in frame.columns
    assert set(frame["data_source"].unique()) == {"train", "test"}

    fig = display.plot()
    assert isinstance(fig.axes[0], mpl.axes.Axes)

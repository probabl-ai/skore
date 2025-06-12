import matplotlib as mpl
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import CrossValidationReport
from skore.sklearn._plot import PrecisionRecallCurveDisplay
from skore.sklearn._plot.utils import sample_mpl_colormap
from skore.utils._testing import check_legend_position, check_precision_recall_frame
from skore.utils._testing import (
    check_precision_recall_curve_display_data as check_display_data,
)


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_binary_classification(
    pyplot, binary_classification_data_no_split, data_source
):
    """Check the attributes and default plotting behaviour of the
    precision-recall curve plot with binary data.
    """
    (estimator, X, y), cv = binary_classification_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    if data_source == "X_y":
        precision_recall_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        precision_recall_kwargs = {"data_source": data_source}

    display = report.metrics.precision_recall(**precision_recall_kwargs)
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    display.plot()

    pos_label = report.estimator_reports_[0].estimator_.classes_[1]

    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == cv

    expected_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for split_idx, line in enumerate(display.lines_):
        assert isinstance(line, mpl.lines.Line2D)
        average_precision = display.average_precision.query(
            f"label == {pos_label} & split_index == {split_idx}"
        )["average_precision"].item()

        assert line.get_label() == (
            f"Fold #{split_idx + 1} (AP = {average_precision:0.2f})"
        )
        assert mpl.colors.to_rgba(line.get_color()) == expected_colors[split_idx]

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    data_source_title = "external" if data_source == "X_y" else data_source
    assert legend.get_title().get_text() == f"{data_source_title.capitalize()} set"
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Recall\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "Precision\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    assert (
        display.ax_.get_title()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
    )


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_multiclass_classification(
    pyplot, multiclass_classification_data_no_split, data_source
):
    """Check the attributes and default plotting behaviour of the precision-recall
    curve plot with multiclass data.
    """
    (estimator, X, y), cv = multiclass_classification_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    if data_source == "X_y":
        precision_recall_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        precision_recall_kwargs = {"data_source": data_source}

    display = report.metrics.precision_recall(**precision_recall_kwargs)
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    display.plot()

    class_labels = report.estimator_reports_[0].estimator_.classes_

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * cv
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(class_labels, default_colors, strict=False):
        for split_idx in range(cv):
            precision_recall_curve_mpl = display.lines_[class_label * cv + split_idx]
            assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
            if split_idx == 0:
                average_precision = display.average_precision.query(
                    f"label == {class_label} & split_index == {split_idx}"
                )["average_precision"]
                assert precision_recall_curve_mpl.get_label() == (
                    f"{str(class_label).title()} "
                    f"(AP = {np.mean(average_precision):0.2f}"
                    f" +/- {np.std(average_precision):0.2f})"
                )
            assert precision_recall_curve_mpl.get_color() == expected_color

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    data_source_title = "external" if data_source == "X_y" else data_source
    assert legend.get_title().get_text() == f"{data_source_title.capitalize()} set"
    assert len(legend.get_texts()) == 3

    assert display.ax_.get_xlabel() == "Recall"
    assert display.ax_.get_ylabel() == "Precision"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    assert (
        display.ax_.get_title()
        == f"Precision-Recall Curve for {estimator.__class__.__name__}"
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["binary_classification_data_no_split", "multiclass_classification_data_no_split"],
)
@pytest.mark.parametrize("pr_curve_kwargs", [[{"color": "red"}], "unknown"])
def test_wrong_kwargs(pyplot, fixture_name, request, pr_curve_kwargs):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `pr_curve_kwargs` argument."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.precision_recall()
    err_msg = (
        "You intend to plot multiple curves. We expect `pr_curve_kwargs` to be a list "
        "of dictionaries"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=pr_curve_kwargs)


def test_frame_binary_classification(binary_classification_data_no_split):
    """Test the frame method with binary classification cross-validation data."""
    (estimator, X, y), cv = binary_classification_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.precision_recall()
    df = display.frame()

    check_precision_recall_frame(
        df,
        expected_n_splits=cv,
        multiclass=False,
    )

    assert df["split_index"].nunique() > 0
    assert df["label"].nunique() == 1
    assert df["estimator_name"].unique() == [report.estimator_name_]


def test_frame_multiclass_classification(multiclass_classification_data_no_split):
    """Test the frame method with multiclass classification cross-validation data."""
    (estimator, X, y), cv = multiclass_classification_data_no_split, 3
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.precision_recall()
    df = display.frame()

    check_precision_recall_frame(
        df,
        expected_n_splits=cv,
        multiclass=True,
    )

    assert df["split_index"].nunique() > 0
    assert df["label"].nunique() == cv
    assert df["estimator_name"].unique() == [report.estimator_name_]


def test_legend(
    pyplot, binary_classification_data_no_split, multiclass_classification_data_no_split
):
    """Check the rendering of the legend for with an `CrossValidationReport`."""

    # binary classification <= 5 folds
    estimator, X, y = binary_classification_data_no_split
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=5)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="lower left", position="inside")

    # binary classification > 5 folds
    estimator, X, y = binary_classification_data_no_split
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=10)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper left", position="outside")

    # multiclass classification <= 5 classes
    estimator, X, y = multiclass_classification_data_no_split
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=5)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="lower left", position="inside")

    # multiclass classification > 5 classes
    estimator = LogisticRegression()
    X, y = make_classification(
        n_samples=1_000,
        n_classes=10,
        n_clusters_per_class=1,
        n_informative=10,
        random_state=42,
    )
    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=10)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper left", position="outside")

import matplotlib as mpl
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import CrossValidationReport
from skore._sklearn._plot import PrecisionRecallCurveDisplay
from skore._sklearn._plot.utils import sample_mpl_colormap
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import (
    check_precision_recall_curve_display_data as check_display_data,
)


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_binary_classification(
    pyplot, logistic_binary_classification_data, data_source
):
    """Check the attributes and default plotting behaviour of the
    precision-recall curve plot with binary data.
    """
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    if data_source == "X_y":
        precision_recall_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        precision_recall_kwargs = {"data_source": data_source}

    display = report.metrics.precision_recall(**precision_recall_kwargs)
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    display.plot()

    pos_label = report.reports_[0].estimator_.classes_[1]

    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == cv

    expected_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for split_idx, line in enumerate(display.lines_):
        assert isinstance(line, mpl.lines.Line2D)
        average_precision = display.average_precision.query(
            f"label == {pos_label} & split == {split_idx}"
        )["average_precision"].item()

        assert line.get_label() == (
            f"Split #{split_idx + 1} (AP = {average_precision:0.2f})"
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
    pyplot, logistic_multiclass_classification_data, data_source
):
    """Check the attributes and default plotting behaviour of the precision-recall
    curve plot with multiclass data.
    """
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    if data_source == "X_y":
        precision_recall_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        precision_recall_kwargs = {"data_source": data_source}

    display = report.metrics.precision_recall(**precision_recall_kwargs)
    assert isinstance(display, PrecisionRecallCurveDisplay)
    check_display_data(display)

    display.plot()

    class_labels = report.reports_[0].estimator_.classes_

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * cv
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(class_labels, default_colors, strict=False):
        for split_idx in range(cv):
            precision_recall_curve_mpl = display.lines_[class_label * cv + split_idx]
            assert isinstance(precision_recall_curve_mpl, mpl.lines.Line2D)
            if split_idx == 0:
                average_precision = display.average_precision.query(
                    f"label == {class_label} & split == {split_idx}"
                )["average_precision"]
                assert precision_recall_curve_mpl.get_label() == (
                    f"{str(class_label).title()} "
                    f"(AP = {np.mean(average_precision):0.2f}"
                    f" +/- {np.std(average_precision, ddof=1):0.2f})"
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
    ["logistic_binary_classification_data", "logistic_multiclass_classification_data"],
)
@pytest.mark.parametrize("pr_curve_kwargs", [[{"color": "red"}], "unknown"])
def test_wrong_kwargs(pyplot, fixture_name, request, pr_curve_kwargs):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `pr_curve_kwargs` argument."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3

    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()
    err_msg = (
        "You intend to plot multiple curves. We expect `pr_curve_kwargs` to be a list "
        "of dictionaries"
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(pr_curve_kwargs=pr_curve_kwargs)


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_binary_classification(
    logistic_binary_classification_data, with_average_precision
):
    """Test the frame method with binary classification data."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    df = report.metrics.precision_recall().frame(
        with_average_precision=with_average_precision
    )
    expected_index = ["split"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["split"].nunique() == cv

    if with_average_precision:
        for (_), group in df.groupby(["split"], observed=True):
            assert group["average_precision"].nunique() == 1


@pytest.mark.parametrize("with_average_precision", [False, True])
def test_frame_multiclass_classification(
    logistic_multiclass_classification_data, with_average_precision
):
    """Test the frame method with multiclass classification data."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    df = report.metrics.precision_recall().frame(
        with_average_precision=with_average_precision
    )
    expected_index = ["split", "label"]
    expected_columns = ["threshold", "precision", "recall"]
    if with_average_precision:
        expected_columns.append("average_precision")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["split"].nunique() == cv
    assert df["label"].nunique() == len(np.unique(y))

    if with_average_precision:
        for (_, _), group in df.groupby(["split", "label"], observed=True):
            assert group["average_precision"].nunique() == 1


def test_legend(
    pyplot, logistic_binary_classification_data, logistic_multiclass_classification_data
):
    """Check the rendering of the legend for with an `CrossValidationReport`."""

    # binary classification <= 5 splits
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="lower left", position="inside")

    # binary classification > 5 splits
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=10)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper left", position="outside")

    # multiclass classification <= 5 classes
    estimator, X, y = logistic_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
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
    report = CrossValidationReport(estimator, X=X, y=y, splitter=10)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_, loc="upper left", position="outside")


def test_binary_classification_constructor(logistic_binary_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()

    index_columns = ["estimator_name", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
        assert df["split"].nunique() == cv
        assert df["label"].unique() == 1

    assert len(display.average_precision) == cv


def test_multiclass_classification_constructor(logistic_multiclass_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()

    index_columns = ["estimator_name", "split", "label"]
    for df in [display.precision_recall, display.average_precision]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator_name"].unique() == report.estimator_name_
        assert df["split"].unique().tolist() == list(range(cv))
        np.testing.assert_array_equal(df["label"].unique(), np.unique(y))

    assert len(display.average_precision) == len(np.unique(y)) * cv

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import CrossValidationReport
from skore._sklearn._plot import PrecisionRecallCurveDisplay
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

    pos_label = report.estimator_reports_[0].estimator_.classes_[1]

    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == cv

    assert isinstance(display.ax_, np.ndarray)
    ax = display.ax_[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]

    average_precision = display.average_precision.query(f"label == {pos_label}")[
        "average_precision"
    ]
    assert (
        f"AP={average_precision.mean():.2f}±{average_precision.std():.2f}"
        in legend_texts
    )

    assert ax.get_xlabel() == "recall"
    assert ax.get_ylabel() == "precision"
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    data_source_title = "external" if data_source == "X_y" else data_source
    suptitle = display.figure_.get_suptitle()
    assert f"Precision-Recall Curve for {estimator.__class__.__name__}" in suptitle
    assert f"Positive label: {pos_label}" in suptitle
    assert f"data source: {data_source_title}" in suptitle.lower()


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

    class_labels = report.estimator_reports_[0].estimator_.classes_

    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * cv

    assert isinstance(display.ax_, np.ndarray)
    assert len(display.ax_) == len(class_labels)

    data_source_title = "external" if data_source == "X_y" else data_source
    for idx in range(len(class_labels)):
        ax = display.ax_[idx]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        assert any("AP=" in text for text in legend_texts)

        assert ax.get_xlabel() == "recall"
        assert ax.get_ylabel() in ("precision", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)

    suptitle = display.figure_.get_suptitle()
    assert f"Precision-Recall Curve for {estimator.__class__.__name__}" in suptitle
    assert f"data source: {data_source_title}" in suptitle.lower()


@pytest.mark.parametrize(
    "fixture_name",
    ["logistic_binary_classification_data", "logistic_multiclass_classification_data"],
)
def test_wrong_kwargs(pyplot, fixture_name, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3

    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.precision_recall()
    with pytest.raises(ValueError, match="subplot_by"):
        display.plot(subplot_by="invalid")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        display.plot(non_existent_kwarg="value")


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
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_[0], loc="upper center", position="inside")

    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=10)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_[0], loc="upper center", position="inside")

    estimator, X, y = logistic_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
    display = report.metrics.precision_recall()
    display.plot()
    check_legend_position(display.ax_[0], loc="upper center", position="inside")

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
    check_legend_position(display.ax_[0], loc="upper center", position="inside")


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

import re

import matplotlib as mpl
import numpy as np
import pytest
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import CrossValidationReport
from skore._sklearn._plot import RocCurveDisplay
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import check_roc_curve_display_data as check_display_data


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_binary_classification(
    pyplot, logistic_binary_classification_data, data_source
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    binary data."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    if data_source == "X_y":
        roc_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        roc_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.roc(**roc_kwargs)
    assert isinstance(display, RocCurveDisplay)

    check_display_data(display)
    pos_label = report.estimator_reports_[0].estimator_.classes_[1]
    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == [pos_label]
        == [display.pos_label]
    )
    assert (
        display.roc_curve["split"].nunique() == display.roc_auc["split"].nunique() == cv
    )

    display.plot()

    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == cv + 1

    expected_color = sns.color_palette()[:1][0]
    for line in display.lines_[:cv]:
        assert line.get_color() == expected_color

    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None

    plot_data = display.frame(with_roc_auc=True)
    roc_auc = plot_data["roc_auc"]
    assert (
        legend.get_texts()[0].get_text()
        == f"AUC={roc_auc.mean():.2f}±{roc_auc.std():.2f}"
    )

    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    data_source_title = "external" if data_source == "X_y" else data_source.capitalize()
    assert (
        display.figure_.get_suptitle()
        == f"ROC Curve for {estimator.__class__.__name__}"
        f"\nPositive label: {pos_label}"
        f"\nData source: {data_source_title} set"
    )


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_multiclass_classification(
    pyplot, logistic_multiclass_classification_data, data_source
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    multiclass data."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    if data_source == "X_y":
        roc_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        roc_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.roc(**roc_kwargs)
    assert isinstance(display, RocCurveDisplay)

    check_display_data(display)
    class_labels = report.estimator_reports_[0].estimator_.classes_
    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == list(class_labels)
    )
    assert (
        display.roc_curve["split"].nunique() == display.roc_auc["split"].nunique() == cv
    )

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * cv + 1

    expected_colors = sns.color_palette()[: len(class_labels)]
    for label_idx in range(len(class_labels)):
        for line in display.lines_[label_idx * cv : (label_idx + 1) * cv]:
            assert line.get_color() == expected_colors[label_idx]

    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]

    for label_idx, class_label in enumerate(class_labels):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {class_label}")["roc_auc"]
        ap_mean = roc_auc.mean()
        ap_std = roc_auc.std()
        assert (
            legend_texts[label_idx] == f"{class_label} (AUC={ap_mean:.2f}±{ap_std:.2f})"
        )

    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)

    data_source_title = "external" if data_source == "X_y" else data_source.capitalize()
    assert (
        display.figure_.get_suptitle()
        == f"ROC Curve for {estimator.__class__.__name__}"
        f"\nData source: {data_source_title} set"
    )


@pytest.mark.parametrize(
    "fixture_name",
    ["logistic_binary_classification_data", "logistic_multiclass_classification_data"],
)
def test_wrong_kwargs(pyplot, fixture_name, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `relplot_kwargs` argument."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3

    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.roc()
    err_msg = "Line2D.set() got an unexpected keyword argument 'invalid'"
    with pytest.raises(AttributeError, match=re.escape(err_msg)):
        display.set_style(relplot_kwargs={"invalid": "value"}).plot()


@pytest.mark.parametrize(
    "fixture_name",
    ["logistic_binary_classification_data", "logistic_multiclass_classification_data"],
)
def test_relplot_kwargs(pyplot, fixture_name, request):
    """Check that we can pass keyword arguments to the ROC curve plot."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.roc()
    multiclass = "multiclass" in fixture_name
    n_labels = (
        len(report.estimator_reports_[0].estimator_.classes_) if multiclass else 1
    )

    display.plot()
    n_roc_lines = n_labels * cv
    default_colors = [line.get_color() for line in display.lines_[:n_roc_lines]]
    if multiclass:
        expected_default = sum([[c] * cv for c in sns.color_palette()[:n_labels]], [])
        assert default_colors == expected_default
    else:
        assert default_colors == [sns.color_palette()[0]] * cv

    if multiclass:
        palette_colors = ["red", "blue", "green"]
        display.set_style(relplot_kwargs={"palette": palette_colors}).plot()
        expected_colors = sum([[c] * cv for c in palette_colors], [])
        for line, expected_color, default_color in zip(
            display.lines_[:n_roc_lines], expected_colors, default_colors, strict=True
        ):
            assert line.get_color() == expected_color
            assert mpl.colors.to_rgb(line.get_color()) != default_color

    else:
        display.set_style(relplot_kwargs={"color": "red"}).plot()
        for line in display.lines_[:n_roc_lines]:
            assert line.get_color() == "red"
            assert mpl.colors.to_rgb(line.get_color()) != default_colors[0]


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_binary_classification(logistic_binary_classification_data, with_roc_auc):
    """Test the frame method with binary classification data."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    df = report.metrics.roc().frame(with_roc_auc=with_roc_auc)
    expected_index = ["split"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["split"].nunique() == cv

    if with_roc_auc:
        for (_), group in df.groupby(["split"], observed=True):
            assert group["roc_auc"].nunique() == 1


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_multiclass_classification(
    logistic_multiclass_classification_data, with_roc_auc
):
    """Test the frame method with multiclass classification data."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    df = report.metrics.roc().frame(with_roc_auc=with_roc_auc)
    expected_index = ["split", "label"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["split"].nunique() == cv
    assert df["label"].nunique() == len(np.unique(y))

    if with_roc_auc:
        for (_, _), group in df.groupby(["split", "label"], observed=True):
            assert group["roc_auc"].nunique() == 1


def test_legend(
    pyplot, logistic_binary_classification_data, logistic_multiclass_classification_data
):
    """Check the rendering of the legend for ROC curves with a
    `CrossValidationReport`."""
    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")

    estimator, X, y = logistic_binary_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=10)
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")

    estimator, X, y = logistic_multiclass_classification_data
    report = CrossValidationReport(estimator, X=X, y=y, splitter=5)
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")

    estimator = LogisticRegression()
    X, y = make_classification(
        n_samples=1_000,
        n_classes=10,
        n_clusters_per_class=1,
        n_informative=10,
        random_state=42,
    )
    report = CrossValidationReport(estimator, X=X, y=y, splitter=10)
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")


def test_binary_classification_constructor(logistic_binary_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_binary_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.roc()

    index_columns = ["estimator", "split", "label"]
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique()[0] == report.estimator_name_
        assert df["split"].nunique() == cv
        assert df["label"].unique() == 1

    assert len(display.roc_auc) == cv


def test_multiclass_classification_constructor(logistic_multiclass_classification_data):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = logistic_multiclass_classification_data, 3
    report = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    display = report.metrics.roc()

    index_columns = ["estimator", "split", "label"]
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique()[0] == report.estimator_name_
        assert df["split"].unique().tolist() == list(range(cv))
        np.testing.assert_array_equal(df["label"].unique(), np.unique(y))

    assert len(display.roc_auc) == len(np.unique(y)) * cv


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        ("logistic_binary_classification_data", ["None", "auto"]),
        ("logistic_multiclass_classification_data", ["None", "auto", "label"]),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X=X, y=y, splitter=3)

    display = report.metrics.roc()
    valid_values_str = ", ".join(valid_values)
    err_msg = f"subplot_by must be one of {valid_values_str}. Got 'invalid' instead."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by, expected_len",
    [
        ("logistic_binary_classification_data", None, 0),
        ("logistic_multiclass_classification_data", "label", 3),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by, expected_len, request):
    """Check that we can pass `None` to `subplot_by`."""
    estimator, X, y = request.getfixturevalue(fixture_name)
    report = CrossValidationReport(estimator, X=X, y=y, splitter=3)
    display = report.metrics.roc()
    display.plot(subplot_by=subplot_by)
    if subplot_by is None:
        assert isinstance(display.ax_, mpl.axes.Axes)
    else:
        assert len(display.ax_) == expected_len

import re

import matplotlib as mpl
import numpy as np
import pytest
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skore import EstimatorReport
from skore._sklearn._plot import RocCurveDisplay
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import check_roc_curve_display_data as check_display_data


def test_binary_classification(pyplot, logistic_binary_classification_with_train_test):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)

    check_display_data(display)
    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == [estimator.classes_[1]]
        == [display.pos_label]
    )

    display.plot()
    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == 1 + 1
    roc_curve_mpl = display.lines_[0]
    assert isinstance(roc_curve_mpl, mpl.lines.Line2D)

    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    plot_data = display.frame(with_roc_auc=True)
    roc_auc = plot_data["roc_auc"].iloc[0]
    assert legend_texts[0] == f"AUC={roc_auc:.2f}"
    expected_color = sns.color_palette()[:1][0]
    assert roc_curve_mpl.get_color() == expected_color

    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() in ("True Positive Rate", "")
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert (
        display.figure_.get_suptitle()
        == f"ROC Curve for {estimator.__class__.__name__}"
        f"\nPositive label: {estimator.classes_[1]}"
        f"\nData source: Test set"
    )


def test_multiclass_classification(
    pyplot, logistic_multiclass_classification_with_train_test
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    multiclass data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)

    check_display_data(display)

    np.testing.assert_array_equal(
        display.roc_curve["label"].unique(), estimator.classes_
    )
    np.testing.assert_array_equal(display.roc_auc["label"].unique(), estimator.classes_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(estimator.classes_) + 1

    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]

    expected_colors = sns.color_palette()[: len(estimator.classes_)]
    for class_label_idx, class_label in enumerate(estimator.classes_):
        roc_curve_mpl = display.lines_[class_label_idx]
        assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {class_label}")["roc_auc"].iloc[0]
        expected_text = f"{class_label} (AUC={roc_auc:.2f})"
        assert legend_texts[class_label_idx] == expected_text
        assert roc_curve_mpl.get_color() == expected_colors[class_label_idx]

    assert len(legend_texts) == len(estimator.classes_) + 1
    assert "Chance level (AUC = 0.5)" in legend_texts
    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"
    assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)

    assert (
        display.figure_.get_suptitle()
        == f"ROC Curve for {estimator.__class__.__name__}"
        f"\nData source: Test set"
    )


def test_data_source(pyplot, logistic_binary_classification_with_train_test):
    """Check that we can pass the `data_source` argument to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="train")
    display.plot()
    ax = display.ax_
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "AUC=1.00" in legend_texts

    display = report.metrics.roc(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    ax = display.ax_
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "AUC=1.00" in legend_texts


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
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    err_msg = "Line2D.set() got an unexpected keyword argument 'invalid'"
    with pytest.raises(AttributeError, match=re.escape(err_msg)):
        display.set_style(relplot_kwargs={"invalid": "value"}).plot()


@pytest.mark.parametrize(
    "fixture_name",
    [
        "logistic_binary_classification_with_train_test",
        "logistic_multiclass_classification_with_train_test",
    ],
)
def test_relplot_kwargs(pyplot, fixture_name, request):
    """Check that we can pass keyword arguments to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    multiclass = "multiclass" in fixture_name
    n_labels = len(estimator.classes_) if multiclass else 1

    display.plot()
    default_colors = [line.get_color() for line in display.lines_[:n_labels]]
    if multiclass:
        expected_default = sns.color_palette()[:n_labels]
        assert default_colors == expected_default
    else:
        assert default_colors == [sns.color_palette()[0]]

    if multiclass:
        palette_colors = ["red", "blue", "green"]
        display.set_style(relplot_kwargs={"palette": palette_colors}).plot()
        expected_colors = palette_colors
    else:
        display.set_style(relplot_kwargs={"color": "red"}).plot()
        expected_colors = ["red"] * n_labels

    for line, expected_color, default_color in zip(
        display.lines_[:n_labels], expected_colors, default_colors, strict=True
    ):
        assert line.get_color() == expected_color
        assert mpl.colors.to_rgb(line.get_color()) != default_color


def test_binary_classification_data_source(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check that we can pass the `data_source` argument to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="train")
    display.plot()
    assert display.ax_.get_legend().get_texts()[0].get_text() == "AUC=1.00"
    assert "Data source: Train set" in display.figure_.get_suptitle()

    display = report.metrics.roc(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert display.ax_.get_legend().get_texts()[0].get_text() == "AUC=1.00"
    assert "Data source: external set" in display.figure_.get_suptitle()


def test_multiclass_classification_data_source(
    pyplot, logistic_multiclass_classification_with_train_test
):
    """Check that we can pass the `data_source` argument to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="train")
    display.plot()
    ax = display.ax_
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    for class_label_idx, class_label in enumerate(estimator.classes_):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {class_label}")["roc_auc"].iloc[0]
        expected_text = f"{class_label} (AUC={roc_auc:.2f})"
        assert legend_texts[class_label_idx] == expected_text
    assert (
        display.figure_.get_suptitle()
        == f"ROC Curve for {estimator.__class__.__name__}"
        f"\nData source: Train set"
    )

    display = report.metrics.roc(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    ax = display.ax_
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    for class_label_idx, class_label in enumerate(estimator.classes_):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"label == {class_label}")["roc_auc"].iloc[0]
        expected_text = f"{class_label} (AUC={roc_auc:.2f})"
        assert legend_texts[class_label_idx] == expected_text
    assert (
        display.figure_.get_suptitle()
        == f"ROC Curve for {estimator.__class__.__name__}"
        f"\nData source: external set"
    )


def test_binary_classification_data_source_both(
    pyplot, logistic_binary_classification_with_train_test
):
    """Check the behavior of the ROC curve plot with binary data
    when data_source='both'.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="both")
    display.plot()
    assert len(display.lines_) == 3
    plot_data = display.frame(with_roc_auc=True)
    roc_auc_train = plot_data.query("data_source == 'train'")["roc_auc"].iloc[0]
    roc_auc_test = plot_data.query("data_source == 'test'")["roc_auc"].iloc[0]
    legend = display.ax_.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert legend_texts[0] == f"Train set (AUC={roc_auc_train:.2f})"
    assert legend_texts[1] == f"Test set (AUC={roc_auc_test:.2f})"


def test_multiclass_classification_data_source_both(
    pyplot, logistic_multiclass_classification_with_train_test
):
    """Check the behavior of the ROC curve plot with multiclass data
    when data_source='both'.
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="both")
    display.plot()

    n_classes = len(estimator.classes_)
    assert len(display.lines_) == n_classes * 2 + 1
    ax = display.ax_
    assert isinstance(ax, mpl.axes.Axes)

    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert len(legend_texts) == n_classes * 2 + 1
    assert "Chance level (AUC = 0.5)" in legend_texts

    for class_label_idx, class_label in enumerate(estimator.classes_):
        plot_data = display.frame(with_roc_auc=True)
        roc_auc_train = plot_data.query(
            f"label == {class_label} & data_source == 'train'"
        )["roc_auc"].iloc[0]
        roc_auc_test = plot_data.query(
            f"label == {class_label} & data_source == 'test'"
        )["roc_auc"].iloc[0]
        train_idx = class_label_idx * 2
        test_idx = class_label_idx * 2 + 1
        train_text = f"{class_label} - Train set (AUC={roc_auc_train:.2f})"
        test_text = f"{class_label} - Test set (AUC={roc_auc_test:.2f})"
        assert legend_texts[train_idx] == train_text
        assert legend_texts[test_idx] == test_text


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_binary_classification(
    logistic_binary_classification_with_train_test, with_roc_auc
):
    """Test the frame method with binary classification data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    df = report.metrics.roc().frame(with_roc_auc=with_roc_auc)
    expected_index = []
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)

    if with_roc_auc:
        assert df["roc_auc"].nunique() == 1


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_multiclass_classification(
    logistic_multiclass_classification_with_train_test, with_roc_auc
):
    """Test the frame method with multiclass classification data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    df = report.metrics.roc().frame(with_roc_auc=with_roc_auc)
    expected_index = ["label"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["label"].nunique() == len(estimator.classes_)

    if with_roc_auc:
        for _, group in df.groupby(["label"], observed=True):
            assert group["roc_auc"].nunique() == 1


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_multiclass_classification_data_source_both(
    logistic_multiclass_classification_with_train_test, with_roc_auc
):
    """
    Test the frame method with multiclass classification data and data_source="both".
    """
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    df = report.metrics.roc(data_source="both").frame(with_roc_auc=with_roc_auc)
    expected_index = ["data_source", "label"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["label"].nunique() == len(estimator.classes_)

    if with_roc_auc:
        for _, group in df.groupby(["label"], observed=True):
            assert group["roc_auc"].nunique() == 2


def test_legend(
    pyplot,
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check the rendering of the legend for ROC curves with an `EstimatorReport`."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")

    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_, loc="upper center", position="inside")


def test_binary_classification_constructor(
    logistic_binary_classification_with_train_test,
):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()

    index_columns = ["estimator", "split", "label"]
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique()[0] == report.estimator_name_
        assert df["split"].isnull().all()
        assert df["label"].unique() == 1

    assert len(display.roc_auc) == 1


def test_multiclass_classification_constructor(
    logistic_multiclass_classification_with_train_test,
):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()

    index_columns = ["estimator", "split", "label"]
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique()[0] == report.estimator_name_
        assert df["split"].isnull().all()
        np.testing.assert_array_equal(df["label"].unique(), estimator.classes_)

    assert len(display.roc_auc) == len(estimator.classes_)


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        ("logistic_binary_classification_with_train_test", ["None", "auto"]),
        (
            "logistic_multiclass_classification_with_train_test",
            ["None", "auto", "label"],
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    err_msg = (
        f"subplot_by must be one of {', '.join(valid_values)}. Got 'invalid' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by, expected_len",
    [
        ("logistic_binary_classification_with_train_test", None, 0),
        ("logistic_multiclass_classification_with_train_test", "label", 3),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by, expected_len, request):
    """Check that we can pass `None` to `subplot_by`."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot(subplot_by=subplot_by)
    if subplot_by is None:
        assert isinstance(display.ax_, mpl.axes.Axes)
    else:
        assert len(display.ax_) == expected_len


def test_plot_chance_level(pyplot, logistic_binary_classification_with_train_test):
    """Check that we can control the display of chance level line."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()

    display.plot(plot_chance_level=True)
    legend = display.ax_.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "Chance level (AUC = 0.5)" in legend_texts
    assert len(legend_texts) == 2

    display.plot(plot_chance_level=False)
    legend = display.ax_.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert "Chance level (AUC = 0.5)" not in legend_texts
    assert len(legend_texts) == 1


def test_despine(pyplot, logistic_binary_classification_with_train_test):
    """Check that despine=True removes the top and right spines."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot(despine=True)
    assert not display.ax_.spines["top"].get_visible()
    assert not display.ax_.spines["right"].get_visible()

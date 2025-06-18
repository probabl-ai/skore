import matplotlib as mpl
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport
from skore.sklearn._plot import RocCurveDisplay
from skore.sklearn._plot.utils import sample_mpl_colormap
from skore.utils._testing import check_frame_structure, check_legend_position
from skore.utils._testing import check_roc_curve_display_data as check_display_data


def test_binary_classification(pyplot, binary_classification_data):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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
    assert len(display.lines_) == 1
    roc_curve_mpl = display.lines_[0]
    assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
    assert (
        roc_curve_mpl.get_label()
        == f"Test set (AUC = {display.roc_auc['roc_auc'].item():0.2f})"
    )
    assert roc_curve_mpl.get_color() == "#1f77b4"  # tab:blue in hex

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == ""
    assert len(legend.get_texts()) == 1 + 1

    assert display.ax_.get_xlabel() == "False Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "True Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    assert display.ax_.get_title() == "ROC Curve for LogisticRegression"


def test_multiclass_classification(pyplot, multiclass_classification_data):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    multiclass data."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
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
    assert hasattr(display, "ax_")
    assert hasattr(display, "figure_")
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(estimator.classes_)
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(
        estimator.classes_, default_colors, strict=False
    ):
        roc_curve_mpl = display.lines_[class_label]
        assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
        roc_auc_class = display.roc_auc.query(f"label == {class_label}")[
            "roc_auc"
        ].item()
        assert roc_curve_mpl.get_label() == (
            f"{str(class_label).title()} (AUC = {roc_auc_class:0.2f})"
        )
        assert roc_curve_mpl.get_color() == expected_color

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Test set"
    assert len(legend.get_texts()) == len(estimator.classes_) + 1

    assert display.ax_.get_xlabel() == "False Positive Rate"
    assert display.ax_.get_ylabel() == "True Positive Rate"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    assert display.ax_.get_title() == "ROC Curve for LogisticRegression"


def test_data_source_binary_classification(pyplot, binary_classification_data):
    """Check that we can pass the `data_source` argument to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="train")
    display.plot()
    assert (
        display.lines_[0].get_label()
        == f"Train set (AUC = {display.roc_auc['roc_auc'].item():0.2f})"
    )

    display = report.metrics.roc(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    assert (
        display.lines_[0].get_label()
        == f"AUC = {display.roc_auc['roc_auc'].item():0.2f}"
    )


def test_data_source_multiclass_classification(pyplot, multiclass_classification_data):
    """Check that we can pass the `data_source` argument to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc(data_source="train")
    display.plot()
    for class_label in estimator.classes_:
        roc_auc_class = display.roc_auc.query(f"label == {class_label}")[
            "roc_auc"
        ].item()
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} (AUC = {roc_auc_class:0.2f})"
        )

    display = report.metrics.roc(data_source="X_y", X=X_train, y=y_train)
    display.plot()
    for class_label in estimator.classes_:
        roc_auc_class = display.roc_auc.query(f"label == {class_label}")[
            "roc_auc"
        ].item()
        assert display.lines_[class_label].get_label() == (
            f"{str(class_label).title()} (AUC = {roc_auc_class:0.2f})"
        )


def test_plot_error_wrong_roc_curve_kwargs(
    pyplot, binary_classification_data, multiclass_classification_data
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    err_msg = (
        "You intend to plot a single curve. We expect `roc_curve_kwargs` to be a "
        "dictionary."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=[{}, {}])

    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    err_msg = "You intend to plot multiple curves."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs={})


@pytest.mark.parametrize("roc_curve_kwargs", [[{"color": "red"}], {"color": "red"}])
def test_roc_curve_kwargs_binary_classification(
    pyplot, binary_classification_data, roc_curve_kwargs
):
    """Check that we can pass keyword arguments to the ROC curve plot."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot(
        roc_curve_kwargs=roc_curve_kwargs, chance_level_kwargs={"color": "blue"}
    )

    assert display.lines_[0].get_color() == "red"
    assert display.chance_level_.get_color() == "blue"

    # check the `.style` display setter
    display.plot()  # default style
    assert display.lines_[0].get_color() == "#1f77b4"
    assert display.chance_level_.get_color() == "k"
    display.set_style(
        roc_curve_kwargs=roc_curve_kwargs, chance_level_kwargs={"color": "blue"}
    )
    display.plot()
    assert display.lines_[0].get_color() == "red"
    assert display.chance_level_.get_color() == "blue"
    # overwrite the style that was set above
    display.plot(
        roc_curve_kwargs={"color": "#1f77b4"}, chance_level_kwargs={"color": "red"}
    )
    assert display.lines_[0].get_color() == "#1f77b4"
    assert display.chance_level_.get_color() == "red"


def test_roc_curve_kwargs_multiclass_classification(
    pyplot, multiclass_classification_data
):
    """Check that we can pass keyword arguments to the ROC curve plot for
    multiclass classification."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot(
        roc_curve_kwargs=[dict(color="red"), dict(color="blue"), dict(color="green")],
        chance_level_kwargs={"color": "blue"},
    )
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[1].get_color() == "blue"
    assert display.lines_[2].get_color() == "green"
    assert display.chance_level_.get_color() == "blue"

    display.plot(plot_chance_level=False)
    assert display.chance_level_ is None

    display.plot(despine=False)
    assert display.ax_.spines["top"].get_visible()
    assert display.ax_.spines["right"].get_visible()


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_binary_classification(binary_classification_data, with_roc_auc):
    """Test the frame method with binary classification data."""
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
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
def test_frame_multiclass_classification(multiclass_classification_data, with_roc_auc):
    """Test the frame method with multiclass classification data."""
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
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
        for (_), group in df.groupby(["label"], observed=True):
            assert group["roc_auc"].nunique() == 1


def test_legend(pyplot, binary_classification_data, multiclass_classification_data):
    """Check the rendering of the legend for ROC curves with an `EstimatorReport`."""

    # binary classification
    estimator, X_train, X_test, y_train, y_test = binary_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_, loc="lower right", position="inside")

    # multiclass classification <= 5 classes
    estimator, X_train, X_test, y_train, y_test = multiclass_classification_data
    report = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_, loc="lower right", position="inside")

    # multiclass classification > 5 classes
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
    check_legend_position(display.ax_, loc="upper left", position="outside")

import matplotlib as mpl
import numpy as np
import pytest
from skore import CrossValidationReport
from skore.sklearn._plot import RocCurveDisplay
from skore.sklearn._plot.utils import sample_mpl_colormap

from .conftest import check_display_data


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_binary_classification(
    pyplot, binary_classification_data_no_split, data_source
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    binary data."""
    (estimator, X, y), cv = binary_classification_data_no_split, 3
    if data_source == "X_y":
        roc_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        roc_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
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
        len(display.roc_curve["split_index"].unique())
        == len(display.roc_auc["split_index"].unique())
        == cv
    )

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == cv
    expected_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for split_idx, line in enumerate(display.lines_):
        assert isinstance(line, mpl.lines.Line2D)
        roc_auc_split = display.roc_auc.query(
            f"label == {pos_label} & split_index == {split_idx}"
        )["roc_auc"].item()
        assert line.get_label() == (
            f"Estimator of fold #{split_idx + 1} (AUC = {roc_auc_split:0.2f})"
        )
        assert mpl.colors.to_rgba(line.get_color()) == expected_colors[split_idx]

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    data_source_title = "external" if data_source == "X_y" else data_source
    assert (
        legend.get_title().get_text()
        == f"LogisticRegression on $\\bf{{{data_source_title}}}$ set"
    )
    assert len(legend.get_texts()) == cv + 1

    assert display.ax_.get_xlabel() == "False Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "True Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


@pytest.mark.parametrize("data_source", ["train", "test", "X_y"])
def test_multiclass_classification(
    pyplot, multiclass_classification_data_no_split, data_source
):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    multiclass data."""
    (estimator, X, y), cv = multiclass_classification_data_no_split, 3
    if data_source == "X_y":
        roc_kwargs = {"data_source": data_source, "X": X, "y": y}
    else:
        roc_kwargs = {"data_source": data_source}

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.roc(**roc_kwargs)
    assert isinstance(display, RocCurveDisplay)

    # check the structure of the attributes
    check_display_data(display)
    class_labels = report.estimator_reports_[0].estimator_.classes_
    assert (
        list(display.roc_curve["label"].unique())
        == list(display.roc_auc["label"].unique())
        == list(class_labels)
    )
    assert (
        len(display.roc_curve["split_index"].unique())
        == len(display.roc_auc["split_index"].unique())
        == cv
    )

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * cv
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for class_label, expected_color in zip(class_labels, default_colors):
        for split_idx in range(cv):
            roc_curve_mpl = display.lines_[class_label * cv + split_idx]
            assert isinstance(roc_curve_mpl, mpl.lines.Line2D)
            if split_idx == 0:
                roc_auc_class = display.roc_auc.query(f"label == {class_label}")[
                    "roc_auc"
                ]
                assert roc_curve_mpl.get_label() == (
                    f"{str(class_label).title()} "
                    f"(AUC = {np.mean(roc_auc_class):0.2f}"
                    f" +/- {np.std(roc_auc_class):0.2f})"
                )
            assert roc_curve_mpl.get_color() == expected_color

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    legend = display.ax_.get_legend()
    data_source_title = "external" if data_source == "X_y" else data_source
    assert (
        legend.get_title().get_text()
        == f"LogisticRegression on $\\bf{{{data_source_title}}}$ set"
    )
    assert len(legend.get_texts()) == cv + 1

    assert display.ax_.get_xlabel() == "False Positive Rate"
    assert display.ax_.get_ylabel() == "True Positive Rate"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)


@pytest.mark.parametrize(
    "roc_curve_kwargs",
    [
        [{"color": "red"}, {"color": "blue"}, {"color": "green"}],
        {"color": "red"},
    ],
)
def test_binary_classification_kwargs(
    pyplot, binary_classification_data_no_split, roc_curve_kwargs
):
    """Check that we can pass keyword arguments to the ROC curve plot for
    cross-validation."""
    (estimator, X, y), cv = binary_classification_data_no_split, 3

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.roc()
    display.plot(roc_curve_kwargs=roc_curve_kwargs)
    if isinstance(roc_curve_kwargs, list):
        assert display.lines_[0].get_color() == "red"
        assert display.lines_[1].get_color() == "blue"
        assert display.lines_[2].get_color() == "green"
    else:
        for line in display.lines_:
            assert line.get_color() == "red"


@pytest.mark.parametrize(
    "fixture_name",
    ["binary_classification_data_no_split", "multiclass_classification_data_no_split"],
)
@pytest.mark.parametrize("roc_curve_kwargs", [[{"color": "red"}], "unknown"])
def test_multiple_roc_curve_kwargs_error(
    pyplot, fixture_name, request, roc_curve_kwargs
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument."""
    (estimator, X, y), cv = request.getfixturevalue(fixture_name), 3

    report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=cv)
    display = report.metrics.roc()
    err_msg = "You intend to plot multiple curves"
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=roc_curve_kwargs)

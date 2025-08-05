"""Test the `roc_curve` display method."""

from itertools import product

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport
from skore._sklearn._plot.metrics.roc_curve import RocCurveDisplay
from skore._sklearn._plot.utils import sample_mpl_colormap
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import check_roc_curve_display_data as check_display_data


@pytest.fixture
def binary_classification_report(binary_classification_data_no_split):
    _, X, y = binary_classification_data_no_split
    estimator_1 = LogisticRegression()
    estimator_2 = LogisticRegression(C=10)
    report = ComparisonReport(
        reports={
            "estimator_1": CrossValidationReport(estimator_1, X, y),
            "estimator_2": CrossValidationReport(estimator_2, X, y),
        }
    )
    return report


@pytest.fixture
def multiclass_classification_report(multiclass_classification_data_no_split):
    _, X, y = multiclass_classification_data_no_split
    estimator_1 = LogisticRegression()
    estimator_2 = LogisticRegression(C=10)
    report = ComparisonReport(
        reports={
            "estimator_1": CrossValidationReport(estimator_1, X, y),
            "estimator_2": CrossValidationReport(estimator_2, X, y),
        }
    )
    return report


def test_binary_classification(pyplot, binary_classification_report):
    """Check the behaviour of `roc_curve` when ML task is "binary-classification"."""
    report = binary_classification_report
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)
    check_display_data(display)

    pos_label = 1
    n_reports = len(report.reports_)
    n_splits = report.reports_[0]._splitter.n_splits

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * n_splits
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for i, estimator_name in enumerate(report.report_names_):
        roc_curve_mpl = display.lines_[i * n_splits]
        assert isinstance(roc_curve_mpl, Line2D)
        auc = display.roc_auc.query(
            f"label == {pos_label} & estimator_name == '{estimator_name}'"
        )["roc_auc"]

        assert roc_curve_mpl.get_label() == (
            f"{report.report_names_[i]} (AUC = {auc.mean():0.2f} +/- {auc.std():0.2f})"
        )
        assert list(roc_curve_mpl.get_color()[:3]) == list(default_colors[i][:3])

    assert isinstance(display.chance_level_, mpl.lines.Line2D)
    assert display.chance_level_.get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_.get_color() == "k"

    assert isinstance(display.ax_, mpl.axes.Axes)
    check_legend_position(display.ax_, loc="lower right", position="inside")
    legend = display.ax_.get_legend()
    assert legend.get_title().get_text() == "Test set"
    assert len(legend.get_texts()) == n_reports + 1

    assert display.ax_.get_xlabel() == "False Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_ylabel() == "True Positive Rate\n(Positive label: 1)"
    assert display.ax_.get_adjustable() == "box"
    assert display.ax_.get_aspect() in ("equal", 1.0)
    assert display.ax_.get_xlim() == display.ax_.get_ylim() == (-0.01, 1.01)
    assert display.ax_.get_title() == "ROC Curve"


def test_multiclass_classification(pyplot, multiclass_classification_report):
    """Check the behaviour of `roc_curve` when ML task is "multiclass-classification"
    and `pos_label` is None."""
    report = multiclass_classification_report
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)
    check_display_data(display)

    labels = display.roc_curve["label"].unique()
    n_reports = len(report.reports_)
    n_splits = report.reports_[0]._splitter.n_splits

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == n_reports * len(labels) * n_splits
    default_colors = sample_mpl_colormap(pyplot.cm.tab10, 10)
    for i, ((estimator_idx, estimator_name), label) in enumerate(
        product(enumerate(report.report_names_), labels)
    ):
        roc_curve_mpl = display.lines_[i * n_splits]
        assert isinstance(roc_curve_mpl, Line2D)

        auc = display.roc_auc.query(
            f"label == {label} & estimator_name == '{estimator_name}'"
        )["roc_auc"]

        assert roc_curve_mpl.get_label() == (
            f"{estimator_name} (AUC = {auc.mean():0.2f} +/- {auc.std():0.2f})"
        )
        assert list(roc_curve_mpl.get_color()[:3]) == list(
            default_colors[estimator_idx][:3]
        )

    assert isinstance(display.chance_level_, list)
    assert isinstance(display.chance_level_[0], mpl.lines.Line2D)
    assert display.chance_level_[0].get_label() == "Chance level (AUC = 0.5)"
    assert display.chance_level_[0].get_color() == "k"

    assert isinstance(display.ax_, np.ndarray)
    for label, ax in zip(labels, display.ax_, strict=False):
        check_legend_position(ax, loc="lower right", position="inside")
        legend = ax.get_legend()
        assert legend.get_title().get_text() == "Test set"
        assert len(legend.get_texts()) == n_reports + 1

        assert ax.get_xlabel() == f"False Positive Rate\n(Positive label: {label})"
        assert ax.get_ylabel() == f"True Positive Rate\n(Positive label: {label})"
        assert ax.get_adjustable() == "box"
        assert ax.get_aspect() in ("equal", 1.0)
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert display.figure_.get_suptitle() == "ROC Curve"


def test_binary_classification_wrong_kwargs(pyplot, binary_classification_report):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument."""
    report = binary_classification_report
    display = report.metrics.roc()
    err_msg = (
        "You intend to plot multiple curves. We expect `roc_curve_kwargs` to be a "
        "list of dictionaries with the same length as the number of curves. "
        "Got 2 instead of 10."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=[{}, {}])


@pytest.mark.parametrize("roc_curve_kwargs", [[{"color": "red"}] * 10])
def test_binary_classification_kwargs(
    pyplot, binary_classification_report, roc_curve_kwargs
):
    """Check that we can pass keyword arguments to the ROC curve plot."""
    report = binary_classification_report
    display = report.metrics.roc()
    display.plot(
        roc_curve_kwargs=roc_curve_kwargs, chance_level_kwargs={"color": "blue"}
    )
    assert display.lines_[0].get_color() == "red"
    assert display.chance_level_.get_color() == "blue"

    # check the `.style` display setter
    display.plot()  # default style
    assert display.lines_[0].get_color() == (
        np.float64(0.12156862745098039),
        np.float64(0.4666666666666667),
        np.float64(0.7058823529411765),
        np.float64(1.0),
    )
    assert display.chance_level_.get_color() == "k"

    display.set_style(
        roc_curve_kwargs=roc_curve_kwargs, chance_level_kwargs={"color": "blue"}
    )
    display.plot()
    assert display.lines_[0].get_color() == "red"
    assert display.chance_level_.get_color() == "blue"

    # overwrite the style that was set above
    display.plot(
        roc_curve_kwargs=[{"color": "#1f77b4"}] * 10,
        chance_level_kwargs={"color": "red"},
    )
    assert display.lines_[0].get_color() == "#1f77b4"
    assert display.chance_level_.get_color() == "red"


def test_multiclass_classification_wrong_kwargs(
    pyplot, multiclass_classification_report
):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `roc_curve_kwargs` argument."""
    report = multiclass_classification_report
    display = report.metrics.roc()
    err_msg = "You intend to plot multiple curves."
    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs=[{}, {}])

    with pytest.raises(ValueError, match=err_msg):
        display.plot(roc_curve_kwargs={})


def test_multiclass_classification_kwargs(pyplot, multiclass_classification_report):
    """Check that we can pass keyword arguments to the ROC curve plot for
    multiclass classification."""
    report = multiclass_classification_report
    display = report.metrics.roc()
    display.plot(
        roc_curve_kwargs=(
            [{"color": "red"}] * 10
            + [{"color": "blue"}] * 10
            + [{"color": "green"}] * 10
        ),
        chance_level_kwargs={"color": "blue"},
    )
    assert display.lines_[0].get_color() == "red"
    assert display.lines_[10].get_color() == "blue"
    assert display.lines_[20].get_color() == "green"
    assert display.chance_level_[0].get_color() == "blue"

    display.plot(plot_chance_level=False)
    assert display.chance_level_ is None

    display.plot(despine=False)
    assert display.ax_[0].spines["top"].get_visible()
    assert display.ax_[0].spines["right"].get_visible()


def test_binary_classification_constructor(binary_classification_data_no_split):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = binary_classification_data_no_split, 3
    report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    # add a different number of splits for the second report
    report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv + 1)
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()

    index_columns = ["estimator_name", "split_index", "label"]
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df.query("estimator_name == 'estimator_1'")[
            "split_index"
        ].unique().tolist() == list(range(cv))
        assert df.query("estimator_name == 'estimator_2'")[
            "split_index"
        ].unique().tolist() == list(range(cv + 1))
        assert df["estimator_name"].unique().tolist() == report.report_names_
        assert df["label"].unique() == 1

    assert len(display.roc_auc) == cv + (cv + 1)


def test_multiclass_classification_constructor(multiclass_classification_data_no_split):
    """Check that the dataframe has the correct structure at initialization."""
    (estimator, X, y), cv = multiclass_classification_data_no_split, 3
    report_1 = CrossValidationReport(estimator, X=X, y=y, splitter=cv)
    report_2 = CrossValidationReport(estimator, X=X, y=y, splitter=cv + 1)
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()

    index_columns = ["estimator_name", "split_index", "label"]
    classes = np.unique(y)
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df.query("estimator_name == 'estimator_1'")[
            "split_index"
        ].unique().tolist() == list(range(cv))
        assert df.query("estimator_name == 'estimator_2'")[
            "split_index"
        ].unique().tolist() == list(range(cv + 1))
        assert df["estimator_name"].unique().tolist() == report.report_names_
        np.testing.assert_array_equal(df["label"].unique(), classes)

    assert len(display.roc_auc) == len(classes) * cv + len(classes) * (cv + 1)


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_binary_classification(binary_classification_report, with_roc_auc):
    """Test the frame method with binary classification comparison
    cross-validation data.
    """
    report = binary_classification_report
    display = report.metrics.roc()
    df = display.frame(with_roc_auc=with_roc_auc)

    expected_index = ["estimator_name", "split_index"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator_name"].nunique() == len(report.reports_)

    if with_roc_auc:
        for (_, _), group in df.groupby(
            ["estimator_name", "split_index"], observed=True
        ):
            assert group["roc_auc"].nunique() == 1


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_multiclass_classification(
    multiclass_classification_report, with_roc_auc
):
    """Test the frame method with multiclass classification comparison
    cross-validation data.
    """
    report = multiclass_classification_report
    display = report.metrics.roc()
    df = display.frame(with_roc_auc=with_roc_auc)

    expected_index = ["estimator_name", "split_index", "label"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator_name"].nunique() == len(report.reports_)

    if with_roc_auc:
        for (_, _, _), group in df.groupby(
            ["estimator_name", "split_index", "label"], observed=True
        ):
            assert group["roc_auc"].nunique() == 1

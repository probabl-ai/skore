import re

import matplotlib as mpl
import numpy as np
import pytest
import seaborn as sns
from sklearn.base import clone

from skore import ComparisonReport, EstimatorReport
from skore._sklearn._plot import RocCurveDisplay
from skore._utils._testing import check_frame_structure, check_legend_position
from skore._utils._testing import check_roc_curve_display_data as check_display_data


def test_binary_classification(pyplot, logistic_binary_classification_with_train_test):
    """Check the attributes and default plotting behaviour of the ROC curve plot with
    binary data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)
    check_display_data(display)
    n_reports = len(report.reports_)

    display.plot()
    assert len(display.ax_) == n_reports

    expected_colors = sns.color_palette()[:1]
    for idx, estimator_name in enumerate(report.reports_):
        ax = display.ax_[idx]
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        plot_data = display.frame(with_roc_auc=True)
        roc_auc = plot_data.query(f"estimator == '{estimator_name}'")["roc_auc"].iloc[0]
        assert legend_texts[0] == f"AUC={roc_auc:.2f}"

        line = display.lines_[idx]
        assert isinstance(line, mpl.lines.Line2D)
        assert line.get_color() == expected_colors[0]

        assert len(legend_texts) == 2
        assert "Chance level (AUC = 0.5)" in legend_texts
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() in ("True Positive Rate", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)
    assert (
        display.figure_.get_suptitle() == f"ROC Curve"
        f"\nPositive label: {display.pos_label}"
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
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.roc()
    assert isinstance(display, RocCurveDisplay)
    check_display_data(display)

    class_labels = next(iter(report.reports_.values())).estimator_.classes_
    n_reports = len(report.reports_)

    display.plot()
    assert isinstance(display.lines_, list)
    assert len(display.lines_) == len(class_labels) * n_reports + n_reports
    expected_colors = sns.color_palette()[: len(class_labels)]
    assert len(display.ax_) == n_reports

    for idx, estimator_name in enumerate(report.reports_):
        ax = display.ax_[idx]
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        for class_label_idx, class_label in enumerate(class_labels):
            plot_data = display.frame(with_roc_auc=True)
            roc_auc = plot_data.query(
                f"label == {class_label} & estimator == '{estimator_name}'"
            )["roc_auc"].iloc[0]
            assert legend_texts[class_label_idx] == f"{class_label} (AUC={roc_auc:.2f})"
            line = ax.get_lines()[class_label_idx]
            assert line.get_color() == expected_colors[class_label_idx]

        assert len(legend_texts) == len(class_labels) + 1
        assert "Chance level (AUC = 0.5)" in legend_texts
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() in ("True Positive Rate", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)

    assert display.figure_.get_suptitle() == "ROC Curve\nData source: Test set"


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

    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
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
    multiclass = "multiclass" in fixture_name
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.roc()
    n_reports = len(report.reports_)
    n_labels = len(display.roc_curve["label"].cat.categories) if multiclass else 1

    display.plot()
    n_roc_lines = n_reports * n_labels
    default_colors = [line.get_color() for line in display.lines_[:n_roc_lines]]
    if multiclass:
        palette_colors = sns.color_palette()[:n_labels]
        expected_default = palette_colors * n_reports
    else:
        expected_default = [sns.color_palette()[0]] * n_reports
    assert default_colors == expected_default

    if multiclass:
        palette_colors = ["red", "blue", "green"]
        display.set_style(relplot_kwargs={"palette": palette_colors}).plot()
        expected_colors = palette_colors * n_reports
    else:
        display.set_style(relplot_kwargs={"color": "red"}).plot()
        expected_colors = ["red"] * n_reports

    for line, expected_color, default_color in zip(
        display.lines_[:n_roc_lines], expected_colors, default_colors, strict=True
    ):
        assert line.get_color() == expected_color
        assert mpl.colors.to_rgb(line.get_color()) != default_color


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_binary_classification(
    logistic_binary_classification_with_train_test, with_roc_auc
):
    """Test the frame method with binary classification comparison data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.roc()
    df = display.frame(with_roc_auc=with_roc_auc)

    expected_index = ["estimator"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == 2

    if with_roc_auc:
        for (_), group in df.groupby(["estimator"], observed=True):
            assert group["roc_auc"].nunique() == 1


@pytest.mark.parametrize("with_roc_auc", [False, True])
def test_frame_multiclass_classification(
    logistic_multiclass_classification_with_train_test, with_roc_auc
):
    """Test the frame method with multiclass classification comparison data."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)
    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )
    display = report.metrics.roc()
    df = display.frame(with_roc_auc=with_roc_auc)

    expected_index = ["estimator", "label"]
    expected_columns = ["threshold", "fpr", "tpr"]
    if with_roc_auc:
        expected_columns.append("roc_auc")

    check_frame_structure(df, expected_index, expected_columns)
    assert df["estimator"].nunique() == 2
    assert df["label"].nunique() == len(estimator.classes_)

    if with_roc_auc:
        for (_, _), group in df.groupby(["estimator", "label"], observed=True):
            assert group["roc_auc"].nunique() == 1


def test_legend(
    pyplot,
    logistic_binary_classification_with_train_test,
    logistic_multiclass_classification_with_train_test,
):
    """Check the rendering of the legend for ROC curves with a `ComparisonReport`."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_[0], loc="upper center", position="inside")

    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()
    display.plot()
    check_legend_position(display.ax_[0], loc="upper center", position="inside")


def test_binary_classification_constructor(
    logistic_binary_classification_with_train_test,
):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()

    index_columns = ["estimator", "split", "label"]
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique().tolist() == list(report.reports_.keys())
        assert df["split"].isnull().all()
        assert df["label"].unique() == 1

    assert len(display.roc_auc) == 2


def test_multiclass_classification_constructor(
    logistic_multiclass_classification_with_train_test,
):
    """Check that the dataframe has the correct structure at initialization."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_multiclass_classification_with_train_test
    )
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()

    index_columns = ["estimator", "split", "label"]
    for df in [display.roc_curve, display.roc_auc]:
        assert all(col in df.columns for col in index_columns)
        assert df["estimator"].unique().tolist() == list(report.reports_.keys())
        assert df["split"].isnull().all()
        np.testing.assert_array_equal(df["label"].unique(), np.unique(y_train))

    assert len(display.roc_auc) == len(np.unique(y_train)) * 2


@pytest.mark.parametrize(
    "fixture_name, valid_values",
    [
        (
            "logistic_binary_classification_with_train_test",
            ["None", "auto", "estimator"],
        ),
        (
            "logistic_multiclass_classification_with_train_test",
            ["auto", "estimator", "label"],
        ),
    ],
)
def test_invalid_subplot_by(fixture_name, valid_values, request):
    """Check that we raise a proper error message when passing an inappropriate
    value for the `subplot_by` argument.
    """
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()
    err_msg = (
        f"subplot_by must be one of {', '.join(valid_values)}. Got 'invalid' instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        display.plot(subplot_by="invalid")


@pytest.mark.parametrize(
    "fixture_name, subplot_by_tuples",
    [
        (
            "logistic_binary_classification_with_train_test",
            [(None, 0), ("estimator", 2)],
        ),
        (
            "logistic_multiclass_classification_with_train_test",
            [("label", 3), ("estimator", 2)],
        ),
    ],
)
def test_valid_subplot_by(fixture_name, subplot_by_tuples, request):
    """Check that we can pass non default values to `subplot_by`."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )
    display = report.metrics.roc()
    for subplot_by, expected_len in subplot_by_tuples:
        display.plot(subplot_by=subplot_by)
        if subplot_by is None:
            assert isinstance(display.ax_, mpl.axes.Axes)
        else:
            assert len(display.ax_) == expected_len


@pytest.mark.parametrize(
    "fixture_name",
    [
        "logistic_binary_classification_with_train_test",
        "logistic_multiclass_classification_with_train_test",
    ],
)
def test_subplot_by_data_source(fixture_name, request):
    """Check the behaviour when `subplot_by` is `data_source`."""
    estimator, X_train, X_test, y_train, y_test = request.getfixturevalue(fixture_name)
    report_1 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report_2 = EstimatorReport(
        estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    report = ComparisonReport(
        reports={"estimator_1": report_1, "estimator_2": report_2}
    )

    display = report.metrics.roc(data_source="both")
    if "multiclass" in fixture_name:
        valid_values = ["auto", "estimator", "label"]
        err_msg = (
            f"subplot_by must be one of {', '.join(valid_values)}. "
            "Got 'data_source' instead."
        )
        with pytest.raises(ValueError, match=err_msg):
            display.plot(subplot_by="data_source")
    else:
        display.plot(subplot_by="data_source")
        assert len(display.ax_) == 2


def test_binary_classification_data_source_both(
    pyplot, logistic_binary_classification_with_train_test
):
    """Regression test: `data_source='both'` should plot without crashing."""
    estimator, X_train, X_test, y_train, y_test = (
        logistic_binary_classification_with_train_test
    )
    estimator_2 = clone(estimator).set_params(C=10).fit(X_train, y_train)

    report = ComparisonReport(
        reports={
            "estimator_1": EstimatorReport(
                estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
            "estimator_2": EstimatorReport(
                estimator_2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            ),
        }
    )

    display = report.metrics.roc(data_source="both")
    assert isinstance(display, RocCurveDisplay)

    display.plot()

    assert isinstance(display.ax_, (list, np.ndarray))
    assert len(display.ax_) == len(report.reports_)

    for ax in display.ax_:
        assert isinstance(ax, mpl.axes.Axes)
        legend = ax.get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert len(legend_texts) == 2 + 1  # 2 datasource + 1 chance level
        assert "Train set" in legend_texts[0]
        assert "Test set" in legend_texts[1]

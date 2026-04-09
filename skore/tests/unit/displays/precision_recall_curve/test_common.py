import numpy as np
import pytest
import seaborn as sns
from sklearn.linear_model import LogisticRegression

from skore import EstimatorReport
from skore._sklearn._plot import PrecisionRecallCurveDisplay
from skore._utils._testing import check_frame_structure


@pytest.mark.parametrize(
    "fixture_prefix",
    [
        "estimator_reports",
        "cross_validation_reports",
        "comparison_estimator_reports",
        "comparison_cross_validation_reports",
    ],
)
class TestPrecisionRecallCurveDisplay:
    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_class_attributes(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.precision_recall()
        assert isinstance(display, PrecisionRecallCurveDisplay)

        assert hasattr(display, "precision_recall")
        assert hasattr(display, "average_precision")
        assert hasattr(display, "report_type")
        assert hasattr(display, "ml_task")
        assert hasattr(display, "data_source")
        assert hasattr(display, "default_pos_label")

        fig = display.plot()
        assert fig is not None
        assert len(fig.axes) >= 1

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    @pytest.mark.parametrize("with_average_precision", [False, True])
    def test_frame_structure(
        self, fixture_prefix, task, with_average_precision, request
    ):
        """Check that the frame method returns a properly structured dataframe."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]

        display = report.metrics.precision_recall()
        frame = display.frame(with_average_precision=with_average_precision)

        expected_columns = ["threshold", "precision", "recall"]
        expected_index = []
        if with_average_precision:
            expected_columns.append("average_precision")
        if "cross_validation" in fixture_prefix:
            expected_index.append("split")
        if "comparison" in fixture_prefix:
            expected_index.append("estimator")
        if task == "multiclass" or display.default_pos_label is None:
            expected_index.append("label")

        check_frame_structure(frame, expected_index, expected_columns)

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_internal_data_structure(self, fixture_prefix, task, request):
        """Check the structure of the precision_recall attribute."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.precision_recall()

        assert list(display.precision_recall.columns) == [
            "estimator",
            "data_source",
            "split",
            "label",
            "threshold",
            "precision",
            "recall",
        ]
        assert list(display.average_precision.columns) == [
            "estimator",
            "data_source",
            "split",
            "label",
            "average_precision",
        ]

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_relplot_kwargs(self, pyplot, fixture_prefix, task, request):
        """Check that heatmap kwargs are applied correctly and can be changed."""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.precision_recall()

        _, ax = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        ax = ax[0]
        assert ax.get_lines()[0].get_color() == sns.color_palette()[0]
        palette = ["red", "green", "blue"] if task == "multiclass" else ["red", "green"]
        relplot_kwargs = {"palette": palette}

        display.set_style(relplot_kwargs=relplot_kwargs)
        fig = display.plot()
        ax = fig.axes[0]
        actual_colors = {line.get_color() for line in ax.get_lines()}
        assert actual_colors == set(palette)

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_plot_structure(self, pyplot, fixture_prefix, task, request):
        """Check that the plot has correct structure"""
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        _, ax = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        ax = ax[0]

        n_splits = 2 if "cross_validation" in fixture_prefix else 1
        n_labels = 3 if task == "multiclass" else 2
        n_lines = n_splits * n_labels
        assert len(ax.get_lines()) == n_lines

        assert ax.get_xlabel() == "recall"
        assert ax.get_ylabel() in ("precision", "")
        assert ax.get_xlim() == ax.get_ylim() == (-0.01, 1.01)

    @pytest.mark.parametrize("task", ["binary", "multiclass"])
    def test_title(self, pyplot, fixture_prefix, task, request):
        report = request.getfixturevalue(f"{fixture_prefix}_{task}_classification")
        if isinstance(report, tuple):
            report = report[0]
        display = report.metrics.precision_recall()
        figure, _ = request.getfixturevalue(
            f"{fixture_prefix}_{task}_classification_figure_axes"
        )
        title = figure.get_suptitle()

        assert "Precision-Recall Curve" in title
        if "comparison" not in fixture_prefix:
            estimator_name = display.precision_recall["estimator"].cat.categories[0]
            assert estimator_name in title
        else:
            assert "for" not in title
        if display.data_source in ("train", "test"):
            assert "Data source" in title
        else:
            assert "Data source" not in title
        if task == "binary" and display.default_pos_label is not None:
            assert "Positive label" in title
        else:
            assert "Positive label" not in title


def test_pos_label(binary_classification_train_test_split):
    """Check that an explicit `pos_label` is reflected by the display."""
    X_train, X_test, y_train, y_test = binary_classification_train_test_split
    labels = np.array(["A", "B"], dtype=object)
    y_train = labels[y_train]
    y_test = labels[y_test]
    estimator = LogisticRegression().fit(X_train, y_train)
    report = EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        pos_label="A",
    )

    display = report.metrics.precision_recall()
    fig = display.plot()

    assert "Positive label: A" in fig.get_suptitle()

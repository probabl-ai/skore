import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from skore.project.widget import ModelExplorerWidget


@pytest.fixture
def metadata():
    """Minimal metadata dataframe containing a regressor and a classifier."""
    metadata = pd.DataFrame(
        data={
            "ml_task": ["classification", "regression", "classification", "regression"],
            "dataset": ["dataset1", "dataset2", "dataset3", "dataset4"],
            "learner": ["learner1", "learner2", "learner3", "learner4"],
            "report_type": [
                "estimator",
                "estimator",
                "cross-validation",
                "cross-validation",
            ],
            "fit_time": [1.0, 2.0, None, None],
            "predict_time": [3.0, 4.0, None, None],
            "rmse": [None, 0.2, None, None],
            "log_loss": [0.3, None, None, None],
            "roc_auc": [0.5, None, None, None],
            # For cross-validation only
            # Mean
            "fit_time_mean": [None, None, 6.0, 7.0],
            "predict_time_mean": [None, None, 8.0, 9.0],
            "rmse_mean": [None, None, None, 10.0],
            "log_loss_mean": [None, None, 11.0, None],
            "roc_auc_mean": [None, None, 12.0, None],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, "id1"), (0, "id2"), (0, "id3"), (0, "id4")], names=[None, "id"]
        ),
    )
    metadata["learner"] = metadata["learner"].astype("category")
    return metadata


def test_model_explorer_widget_check_dataframe_schema_error():
    """Check that we raise an error if the dataframe is missing required columns or
    index."""
    df = pd.DataFrame(
        columns=["xxx"],
        index=pd.MultiIndex.from_tuples(
            [(0, "id1")], names=ModelExplorerWidget._required_index
        ),
    )
    err_msg = re.escape(
        "Dataframe is missing required columns: ['ml_task', 'dataset', 'learner', "
        "'fit_time', 'predict_time', 'rmse', 'log_loss', 'roc_auc', "
        "'fit_time_mean', 'predict_time_mean', 'rmse_mean', 'log_loss_mean', "
        "'roc_auc_mean']"
    )
    with pytest.raises(ValueError, match=err_msg):
        ModelExplorerWidget(df)

    df = pd.DataFrame(columns=ModelExplorerWidget._required_columns, index=["xxx"])
    err_msg = re.escape("Dataframe is missing required index: [None, 'id']")
    with pytest.raises(ValueError, match=err_msg):
        ModelExplorerWidget(df)

    df = pd.DataFrame(
        columns=ModelExplorerWidget._required_columns,
        index=pd.MultiIndex.from_tuples(
            [(0, "id1")], names=ModelExplorerWidget._required_index
        ),
    )
    df["learner"] = "xxx"
    err_msg = re.escape("Learner column must be a categorical column")
    with pytest.raises(ValueError, match=err_msg):
        ModelExplorerWidget(df)


def test_model_explorer_widget_empty_dataframe(capsys):
    """Check that we can pass an empty dataframe but the HTML representation is
    bypassed."""
    df = pd.DataFrame([])
    widget = ModelExplorerWidget(df)

    assert widget.dataframe.empty
    assert widget.seed == 0

    widget.display()

    captured = capsys.readouterr()
    assert "No report found in the project. Use the `put` method to add reports." in (
        captured.out
    )


def test_model_explorer_widget_controls(metadata):
    """Check that the controls are correctly populated."""
    widget = ModelExplorerWidget(metadata)
    np.testing.assert_array_equal(
        widget._datasets("classification", "estimator"), ["dataset1"]
    )
    np.testing.assert_array_equal(
        widget._datasets("regression", "estimator"), ["dataset2"]
    )
    np.testing.assert_array_equal(
        widget._datasets("classification", "cross-validation"), ["dataset3"]
    )
    np.testing.assert_array_equal(
        widget._datasets("regression", "cross-validation"), ["dataset4"]
    )
    assert widget._metric_checkboxes.keys() == {"classification", "regression"}
    assert widget._metric_checkboxes["classification"].keys() == {
        "fit_time",
        "predict_time",
        "log_loss",
        "roc_auc",
    }
    assert widget._metric_checkboxes["regression"].keys() == {
        "fit_time",
        "predict_time",
        "rmse",
    }
    assert widget._color_metric_dropdown.keys() == {"classification", "regression"}
    assert widget._color_metric_dropdown["classification"].options == (
        "Fit Time",
        "Predict Time",
        "Log Loss",
        "Macro ROC AUC",
    )
    assert widget._color_metric_dropdown["regression"].options == (
        "Fit Time",
        "Predict Time",
        "RMSE",
    )
    assert len(widget.classification_metrics_box.children) == 2
    # time
    components = widget.classification_metrics_box.children[0].children
    assert len(components) == 3
    assert components[0].value == "Computation Metrics: "
    assert components[1].description == "Fit Time"
    assert components[2].description == "Predict Time"
    # statistical
    components = widget.classification_metrics_box.children[1].children
    assert len(components) == 3
    assert components[0].value == "Statistical Metrics: "
    assert components[1].description == "Macro ROC AUC"
    assert components[2].description == "Log Loss"

    # regression
    components = widget.regression_metrics_box.children[0].children
    assert len(components) == 3
    assert components[0].value == "Computation Metrics: "
    assert components[1].description == "Fit Time"
    assert components[2].description == "Predict Time"
    # statistical
    components = widget.regression_metrics_box.children[1].children
    assert len(components) == 2
    assert components[0].value == "Statistical Metrics: "
    assert components[1].description == "RMSE"

    assert widget.current_fig is None
    assert widget.current_selection == {}


def test_model_explorer_widget_single_task(metadata):
    """Check the behaviour of the widget when there is a single task in the metadata.

    We switch ML task to the one having at least an item.
    """
    metadata = metadata.copy()  # do not modify the fixture
    metadata["ml_task"] = "classification"
    widget = ModelExplorerWidget(metadata)
    # check that the classification dropdown menu is visible
    assert widget.classification_metrics_box.layout.display == ""
    assert widget._color_metric_dropdown["classification"].layout.display == ""
    # check that the regression dropdown menu is hidden
    assert widget.regression_metrics_box.layout.display == "none"
    assert widget._color_metric_dropdown["regression"].layout.display == "none"

    metadata["ml_task"] = "regression"
    widget = ModelExplorerWidget(metadata)
    # check that the classification dropdown menu is hidden
    assert widget.classification_metrics_box.layout.display == "none"
    assert widget._color_metric_dropdown["classification"].layout.display == "none"
    # check that the regression dropdown menu is visible
    assert widget.regression_metrics_box.layout.display == ""
    assert widget._color_metric_dropdown["regression"].layout.display == ""


def test_model_explorer_widget_jitter(metadata):
    """Check the behaviour of the static method adding some jitter for categorical
    feature represented on the parallel coordinate plot."""
    widget = ModelExplorerWidget(metadata)
    amount = 0.01
    jitted_categories = widget._add_jitter_to_categorical(
        seed=0, categorical_series=widget.dataframe["learner"], amount=amount
    )
    assert amount > jitted_categories[0] > 0
    assert 1 > jitted_categories[1] > 1 - amount


def test_model_explorer_widget_update_plot(metadata):
    """Check the behaviour of the method `_update_plot` and thus programmatically
    if the information on the parallel coordinate plot are the one expected."""
    widget = ModelExplorerWidget(metadata)
    widget._update_plot()

    assert isinstance(widget.current_fig, go.FigureWidget)
    parallel_coordinate = widget.current_fig.data[0]
    # by default with the data at hand we expect a classification problem
    dimension_labels = [dim["label"] for dim in parallel_coordinate.dimensions]
    assert dimension_labels == ["Learner", "Log Loss", "Macro ROC AUC"]
    assert parallel_coordinate.dimensions[0]["ticktext"] == ["learner1"]
    np.testing.assert_array_equal(
        parallel_coordinate.dimensions[1]["values"],
        metadata["log_loss"].dropna().to_numpy(),
    )
    np.testing.assert_array_equal(
        parallel_coordinate.dimensions[2]["values"],
        metadata["roc_auc"].dropna().to_numpy(),
    )
    assert parallel_coordinate["line"]["colorbar"]["title"]["text"] == "Log Loss"

    # simulate a change to the regression task
    widget._task_dropdown.value = "regression"
    widget._update_plot()
    assert isinstance(widget.current_fig, go.FigureWidget)
    parallel_coordinate = widget.current_fig.data[0]
    dimension_labels = [dim["label"] for dim in parallel_coordinate.dimensions]
    assert dimension_labels == ["Learner", "RMSE"]
    assert parallel_coordinate.dimensions[0]["ticktext"] == ["learner2"]
    np.testing.assert_array_equal(
        parallel_coordinate.dimensions[1]["values"],
        metadata["rmse"].dropna().to_numpy(),
    )
    assert parallel_coordinate["line"]["colorbar"]["title"]["text"] == "RMSE"

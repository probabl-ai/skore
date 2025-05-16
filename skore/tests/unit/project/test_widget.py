import re

import numpy as np
import pandas as pd
import pytest
from skore.project.widget import ModelExplorerWidget


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
        "'fit_time', 'predict_time', 'rmse', 'log_loss', 'roc_auc']"
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
    assert "No reports found in the project. Use the `put` method to add reports." in (
        captured.out
    )


def test_model_explorer_widget_controls():
    """Check that the controls are correctly populated."""
    metadata = pd.DataFrame(
        data={
            "ml_task": ["classification", "regression"],
            "dataset": ["dataset1", "dataset2"],
            "learner": ["learner1", "learner2"],
            "fit_time": [1.0, 2.0],
            "predict_time": [3.0, 4.0],
            "rmse": [None, 0.2],
            "log_loss": [0.3, None],
            "roc_auc": [0.5, None],
        },
        index=pd.MultiIndex.from_tuples([(0, "id1"), (0, "id2")], names=[None, "id"]),
    )
    metadata["learner"] = metadata["learner"].astype("category")

    widget = ModelExplorerWidget(metadata)
    np.testing.assert_array_equal(widget._clf_datasets, ["dataset1"])
    np.testing.assert_array_equal(widget._reg_datasets, ["dataset2"])
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


def test_model_explorer_widget_single_task():
    """Check the behaviour of the widget when there is a single task in the metadata.

    We switch ML task to the one having at least an item.
    """
    metadata = pd.DataFrame(
        data={
            "ml_task": ["classification"],
            "dataset": ["dataset1"],
            "learner": ["learner1"],
            "fit_time": [1.0],
            "predict_time": [2.0],
            "rmse": [0.1],
            "log_loss": [0.2],
            "roc_auc": [0.3],
        },
        index=pd.MultiIndex.from_tuples([(0, "id1")], names=[None, "id"]),
    )
    metadata["learner"] = metadata["learner"].astype("category")

    widget = ModelExplorerWidget(metadata)
    # check that the classification dropdown menu is visible
    assert widget.classification_metrics_box.layout.display == ""
    assert widget._color_metric_dropdown["classification"].layout.display == ""
    # check that the regression dropdown menu is hidden
    assert widget.regression_metrics_box.layout.display == "none"
    assert widget._color_metric_dropdown["regression"].layout.display == "none"

    metadata["ml_task"] = ["regression"]

    widget = ModelExplorerWidget(metadata)
    # check that the classification dropdown menu is hidden
    assert widget.classification_metrics_box.layout.display == "none"
    assert widget._color_metric_dropdown["classification"].layout.display == "none"
    # check that the regression dropdown menu is visible
    assert widget.regression_metrics_box.layout.display == ""
    assert widget._color_metric_dropdown["regression"].layout.display == ""

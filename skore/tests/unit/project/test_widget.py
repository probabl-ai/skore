import re

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

"""Real local-project sync.

Lives beside the heavier plugin tests so collecting the rest of the CLI suite
does not import scikit-learn or construct a skore project.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from skore._cli.sync.commands import sync


def test_sync_transfers_between_real_local_projects(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("skore")
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    from skore import Project, evaluate

    source_workspace = tmp_path / "source"
    destination_workspace = tmp_path / "destination"
    X, y = make_regression(n_samples=20, n_features=2, random_state=0)
    source = Project("source", mode="local", workspace=source_workspace)
    source.put("model", evaluate(LinearRegression(), X, y, splitter=0.2))

    result = CliRunner().invoke(
        sync,
        [
            "source",
            "--from=local",
            f"--from-workspace={source_workspace}",
            "--to=local",
            "--to-project=destination",
            f"--to-workspace={destination_workspace}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "transferred" in result.output
    destination = Project("destination", mode="local", workspace=destination_workspace)
    assert destination.summarize().frame()["key"].tolist() == ["model"]

import os

import pytest
from skore.cli.cli import cli


@pytest.fixture
def fake_launch(monkeypatch):
    def _fake_launch(project_name, port, open_browser, verbose):
        pass

    monkeypatch.setattr("skore.cli.quickstart_command.__launch", _fake_launch)


def test_quickstart(tmp_path, fake_launch):
    os.chdir(tmp_path)
    cli("quickstart".split())
    assert (tmp_path / "project.skore").exists()

    # calling the same command without overwriting should succeed
    # (as the creation step is skipped)
    cli("quickstart".split())

    # calling the same command with overwriting should succeed
    cli("quickstart --overwrite".split())

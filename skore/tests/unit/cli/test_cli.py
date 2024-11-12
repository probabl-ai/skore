"""Test CLI properly calls the app."""

import pytest
from skore.cli.cli import cli


def test_cli_launch(monkeypatch):
    launch_project_name = None
    launch_port = None
    launch_open_browser = None

    def fake_launch(project_name, port, open_browser):
        nonlocal launch_project_name
        nonlocal launch_port
        nonlocal launch_open_browser

        launch_project_name = project_name
        launch_port = port
        launch_open_browser = open_browser

    monkeypatch.setattr("skore.cli.cli.__launch", fake_launch)
    monkeypatch.setattr("sys.argv", ["skore", "launch", "project.skore", "--port", "0", "--no-open-browser"])

    cli()

    assert launch_project_name == "project.skore"
    assert launch_port == 0
    assert not launch_open_browser


def test_cli_launch_no_project_name(monkeypatch):
    monkeypatch.setattr("sys.argv", ["skore", "launch", "--port", 0, "--no-open-browser"])

    with pytest.raises(SystemExit):
        cli()

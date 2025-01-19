"""Test CLI properly calls the app."""

import pytest
from skore.cli.cli import cli


def test_cli_launch(monkeypatch):
    launch_project_name = None
    launch_port = None
    launch_open_browser = None
    launch_verbose = None

    def fake_launch(project_name, port, open_browser, verbose):
        nonlocal launch_project_name
        nonlocal launch_port
        nonlocal launch_open_browser
        nonlocal launch_verbose

        launch_project_name = project_name
        launch_port = port
        launch_open_browser = open_browser
        launch_verbose = verbose

    monkeypatch.setattr("skore.cli.cli._launch", fake_launch)
    cli(["launch", "project.skore", "--port", "0", "--no-open-browser", "--verbose"])

    assert launch_project_name == "project.skore"
    assert launch_port == 0
    assert not launch_open_browser
    assert launch_verbose


def test_cli_launch_no_project_name():
    with pytest.raises(SystemExit):
        cli(["launch", "--port", 0, "--no-open-browser", "--verbose"])

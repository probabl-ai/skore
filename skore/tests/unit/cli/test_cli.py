import pytest
from skore.cli.cli import cli


def test_cli(monkeypatch, tmp_path):
    """cli passes its arguments down to `launch`."""

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

    monkeypatch.setattr("skore.cli.cli.launch", fake_launch)

    cli(
        [
            str(tmp_path / "my_project.skore"),
            "--port",
            "888",
            "--no-open-browser",
            "--verbose",
        ]
    )

    assert launch_project_name == str(tmp_path / "my_project.skore")
    assert launch_port == 888
    assert launch_open_browser is False
    assert launch_verbose is True


def test_cli_no_project_name():
    with pytest.raises(SystemExit):
        cli([])

from pathlib import Path

from skore.cli.cli import cli


def test_quickstart(monkeypatch):
    """`quickstart` passes its arguments down to `create` and `launch`."""

    create_project_name = None
    create_working_dir = None
    create_overwrite = None

    def fake_create(project_name, working_dir, overwrite):
        nonlocal create_project_name
        nonlocal create_working_dir
        nonlocal create_overwrite

        create_project_name = project_name
        create_working_dir = working_dir
        create_overwrite = overwrite

    monkeypatch.setattr("skore.cli.quickstart_command.create", fake_create)

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

    monkeypatch.setattr("skore.cli.quickstart_command.__launch", fake_launch)

    cli(
        [
            "quickstart",
            "my_project.skore",
            "--overwrite",
            "--working-dir",
            "hello",
            "--port",
            "888",
            "--no-open-browser",
        ]
    )

    assert create_project_name == "my_project.skore"
    assert create_working_dir == Path("hello")
    assert create_overwrite is True

    assert launch_project_name == "my_project.skore"
    assert launch_port == 888
    assert launch_open_browser is False

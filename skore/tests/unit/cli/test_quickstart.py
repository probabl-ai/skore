from skore.cli.cli import cli


def test_quickstart(monkeypatch, tmp_path):
    """`quickstart` passes its arguments down to `create` and `launch`."""

    create_path = None
    create_create = None
    create_overwrite = None

    class FakeProject:
        def __init__(self, path, create, overwrite):
            nonlocal create_path
            nonlocal create_create
            nonlocal create_overwrite

            create_path = path
            create_create = create
            create_overwrite = overwrite

    monkeypatch.setattr("skore.cli.cli.Project", FakeProject)

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

    monkeypatch.setattr("skore.cli.cli.__launch", fake_launch)

    cli(
        [
            "quickstart",
            str(tmp_path / "my_project.skore"),
            "--verbose",
            "--overwrite",
            "--port",
            "888",
            "--no-open-browser",
        ]
    )

    assert create_path == str(tmp_path / "my_project.skore")
    assert create_create is True
    assert create_overwrite is True

    assert launch_project_name == str(tmp_path / "my_project.skore")
    assert launch_port == 888
    assert launch_open_browser is False
    assert launch_verbose is True

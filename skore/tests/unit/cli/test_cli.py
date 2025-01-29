import pytest
from skore.cli.cli import cli


@pytest.mark.parametrize(
    "args,expected_outputs",
    [
        (["--help"], ["usage: skore-ui", "open", "kill"]),
        (["-h"], ["usage: skore-ui", "open", "kill"]),
        (["open", "--help"], ["usage: skore-ui open", "project_path", "--serve"]),
        (["open", "-h"], ["usage: skore-ui open", "project_path", "--serve"]),
    ],
)
def test_help_messages(capsys, args, expected_outputs):
    """Test that help messages are displayed correctly."""
    with pytest.raises(SystemExit) as exc_info:
        cli(args)

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    for expected in expected_outputs:
        assert expected in captured.out


def test_kill_command(tmp_path, monkeypatch):
    """Test that kill command executes without errors and calls the right function."""
    # Mock platformdirs state dir to use a temporary directory and avoid side-effects
    with monkeypatch.context() as m:
        m.setattr("platformdirs.user_state_path", lambda appname: tmp_path)
        cli(["kill"])


def test_open_command(tmp_path, monkeypatch):
    """Test that open command."""
    project_path = tmp_path / "test_project.skore"
    project_path.mkdir()

    # Mock _launch to prevent browser opening and match the actual signature
    with monkeypatch.context() as m:
        m.setattr(
            "skore.project._launch._launch",
            lambda project,
            keep_alive="auto",
            port=None,
            open_browser=True,
            verbose=False: None,
        )

        cli(["open", str(project_path), "--no-serve"])

        assert project_path.exists()
        assert project_path.is_dir()

        # We should be able to open the same project again
        cli(["open", str(project_path), "--no-serve"])

        assert project_path.exists()
        assert project_path.is_dir()

        # Let's start the server
        cli(["open", str(project_path), "--serve"])

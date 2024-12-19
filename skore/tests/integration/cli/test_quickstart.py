from skore.cli.cli import cli


def test_quickstart(tmp_path, monkeypatch):
    def fake_launch(project_name, port, open_browser, verbose):
        pass

    monkeypatch.setattr("skore.cli.quickstart_command.__launch", fake_launch)

    cli(f"quickstart --working-dir {tmp_path}".split())
    assert (tmp_path / "project.skore").exists()

    # calling the same command without overwriting should succeed
    # (as the creation step is skipped)
    cli(f"quickstart --working-dir {tmp_path}".split())

    # calling the same command with overwriting should succeed
    cli(f"quickstart --working-dir {tmp_path} --overwrite".split())

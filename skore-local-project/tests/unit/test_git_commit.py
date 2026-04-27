from skore_local_project.metadata import git_commit


def test_happy_path(monkeypatch):
    monkeypatch.setattr("skore_local_project.metadata.working_tree_clean", lambda: True)
    monkeypatch.setattr(
        "skore_local_project.metadata.head_commit_hash", lambda: "abc123"
    )
    assert git_commit() == "abc123"


def test_working_tree_dirty(monkeypatch):
    monkeypatch.setattr(
        "skore_local_project.metadata.working_tree_clean", lambda: False
    )
    assert git_commit() is None


def test_git_unavailable(monkeypatch):
    def raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("subprocess.run", raise_file_not_found)

    assert git_commit() is None

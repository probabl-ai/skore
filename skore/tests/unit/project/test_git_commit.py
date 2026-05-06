import subprocess

import pytest

from skore._project.git import git_commit, head_commit_hash, working_tree_clean


def success(*, stdout):
    """Emulate a completed process."""

    def func(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    return func


def raise_CalledProcessError(*args, **kwargs):
    raise subprocess.CalledProcessError(128, "git")


class TestWorkingTreeClean:
    def test_clean(self, monkeypatch):
        monkeypatch.setattr("subprocess.run", success(stdout=""))

        assert working_tree_clean() is True

    def test_dirty(self, monkeypatch):
        monkeypatch.setattr("subprocess.run", success(stdout=" M file.py\n"))

        assert working_tree_clean() is False

    def test_git_error(self, monkeypatch):
        monkeypatch.setattr("subprocess.run", raise_CalledProcessError)

        assert working_tree_clean() is False


class TestHeadCommitHash:
    def test_success(self, monkeypatch):
        monkeypatch.setattr("subprocess.run", success(stdout="abc123\n"))

        assert head_commit_hash() == "abc123"

    def test_git_error(self, monkeypatch):
        monkeypatch.setattr("subprocess.run", raise_CalledProcessError)

        assert head_commit_hash() is None


class TestGitCommit:
    @pytest.fixture(autouse=True)
    def clear_git_commit_cache(self):
        git_commit.cache_clear()

    def test_clean_tree(self, monkeypatch):
        monkeypatch.setattr("skore._project.git.working_tree_clean", lambda: True)
        monkeypatch.setattr("skore._project.git.head_commit_hash", lambda: "abc123")
        monkeypatch.setattr("skore._project.git.git_available", lambda: True)

        assert git_commit() == "abc123"

    def test_dirty_tree(self, monkeypatch):
        monkeypatch.setattr("skore._project.git.working_tree_clean", lambda: False)
        monkeypatch.setattr("skore._project.git.head_commit_hash", lambda: "abc123")
        monkeypatch.setattr("skore._project.git.git_available", lambda: True)

        assert git_commit() == "abc123 (working tree dirty)"

    def test_git_unavailable(self, monkeypatch):
        monkeypatch.setattr("skore._project.git.git_available", lambda: False)

        assert git_commit() is None

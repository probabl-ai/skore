import os
from contextlib import contextmanager

import pytest
from skore.project import Project, open


@pytest.fixture
def tmp_project_path(tmp_path):
    """Create a project at `tmp_path` and return its absolute path."""
    # Project path must end with ".skore"
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    os.mkdir(project_path / "items")
    os.mkdir(project_path / "views")
    return project_path


@contextmanager
def project_changed(project_path, modified=True):
    """Assert that the project at `project_path` was changed.

    If `modified` is False, instead assert that it was *not* changed.
    """
    (project_path / "my_test_file").write_text("hello")
    yield
    if modified:
        assert not (project_path / "my_test_file").exists()
    else:
        assert (project_path / "my_test_file").exists()


def test_open_relative_path(tmp_project_path):
    """If passed a relative path, `open` operates in the current working directory."""
    os.chdir(tmp_project_path.parent)
    p = open(tmp_project_path.name, create=False)
    assert isinstance(p, Project)


def test_open_default(tmp_project_path):
    """If a project already exists, `open` loads it."""
    with project_changed(tmp_project_path, modified=False):
        p = open(tmp_project_path)
        assert isinstance(p, Project)


def test_open_default_no_project(tmp_path):
    """If no project exists, `open` creates it."""
    with project_changed(tmp_path, modified=False):
        p = open(tmp_path)
        assert isinstance(p, Project)


def test_open_project_exists_create_true_overwrite_true(tmp_project_path):
    """If the project exists, and `create` and `overwrite` are set to `True`,
    `open` overwrites it with a new one."""
    with project_changed(tmp_project_path):
        open(tmp_project_path, create=True, overwrite=True)


def test_open_project_exists_create_true_overwrite_false(tmp_project_path):
    with project_changed(tmp_project_path, modified=False):
        open(tmp_project_path, create=True, overwrite=False)


def test_open_project_exists_create_false_overwrite_true(tmp_project_path):
    p = open(tmp_project_path, create=False, overwrite=True)
    assert isinstance(p, Project)


def test_open_project_exists_create_false_overwrite_false(tmp_project_path):
    p = open(tmp_project_path, create=False, overwrite=False)
    assert isinstance(p, Project)


def test_open_no_project_create_true_overwrite_true(tmp_path):
    p = open(tmp_path, create=True, overwrite=True)
    assert isinstance(p, Project)


def test_open_no_project_create_true_overwrite_false(tmp_path):
    p = open(tmp_path, create=True, overwrite=False)
    assert isinstance(p, Project)


def test_open_no_project_create_false_overwrite_true(tmp_path):
    with pytest.raises(FileNotFoundError):
        open(tmp_path, create=False, overwrite=True)


def test_open_no_project_create_false_overwrite_false(tmp_path):
    with pytest.raises(FileNotFoundError):
        open(tmp_path, create=False, overwrite=False)

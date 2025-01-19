import os

import pytest
from skore.project import Project
from skore.project._load import load


@pytest.fixture
def tmp_project_path(tmp_path):
    """Create a project at `tmp_path` and return its absolute path."""
    # Project path must end with ".skore"
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    os.mkdir(project_path / "items")
    os.mkdir(project_path / "views")
    return project_path


def test_load_no_project():
    with pytest.raises(FileNotFoundError):
        load("/empty")


def test_load_absolute_path(tmp_project_path):
    p = load(tmp_project_path)
    assert isinstance(p, Project)


def test_load_relative_path(tmp_project_path):
    os.chdir(tmp_project_path.parent)
    p = load(tmp_project_path.name)
    assert isinstance(p, Project)

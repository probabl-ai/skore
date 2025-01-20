import os

import pytest
from skore.exceptions import ProjectLoadError
from skore.project import Project
from skore.project._load import _load


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
        _load("/empty")


def test_load_absolute_path(tmp_project_path):
    p = _load(tmp_project_path)
    assert isinstance(p, Project)


def test_load_relative_path(tmp_project_path):
    os.chdir(tmp_project_path.parent)
    p = _load(tmp_project_path.name)
    assert isinstance(p, Project)


def test_load_corrupted_project(tmp_path):
    """Test loading a project with missing subdirectories raises ProjectLoadError."""
    # Create project directory without required subdirectories
    project_path = tmp_path.parent / (tmp_path.name + ".skore")
    os.mkdir(project_path)
    # Only create 'items' directory, leaving 'views' missing
    os.mkdir(project_path / "items")

    with pytest.raises(ProjectLoadError) as exc_info:
        _load(project_path)

    assert "Project" in str(exc_info.value)
    assert "corrupted" in str(exc_info.value)
    assert "views" in str(exc_info.value)
    assert "Consider re-creating the project" in str(exc_info.value)

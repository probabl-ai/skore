from pathlib import Path

import pytest
from skore.exceptions import (
    InvalidProjectNameError,
    ProjectCreationError,
    ProjectPermissionError,
)
from skore.project._create import _create, _validate_project_name


@pytest.mark.parametrize(
    "project_name,expected",
    [
        (
            "a" * 250,
            (False, InvalidProjectNameError()),
        ),
        (
            "%",
            (False, InvalidProjectNameError()),
        ),
        (
            "hello world",
            (False, InvalidProjectNameError()),
        ),
    ],
)
def test_validate_project_name(project_name, expected):
    result, exception = _validate_project_name(project_name)
    expected_result, expected_exception = expected
    assert result == expected_result
    assert type(exception) is type(expected_exception)


@pytest.mark.parametrize("project_name", ["hello", "hello.skore"])
def test_create_project(project_name, tmp_path):
    _create(tmp_path / project_name)
    assert (tmp_path / "hello.skore").exists()


# TODO: If using fixtures in test cases is possible, join this with
# `test_create_project`
def test_create_project_absolute_path(tmp_path):
    _create(tmp_path / "hello")
    assert (tmp_path / "hello.skore").exists()


def test_create_project_fails_if_file_exists(tmp_path):
    _create(tmp_path / "hello")
    assert (tmp_path / "hello.skore").exists()
    with pytest.raises(FileExistsError):
        _create(tmp_path / "hello")


def test_create_project_fails_if_permission_denied(tmp_path):
    with pytest.raises(ProjectCreationError):
        _create("/")


@pytest.mark.parametrize("project_name", ["hello.txt", "%%%", "COM1"])
def test_create_project_fails_if_invalid_name(project_name, tmp_path):
    with pytest.raises(ProjectCreationError):
        _create(tmp_path / project_name)


def test_create_project_invalid_names():
    """Test project creation with invalid names."""
    invalid_names = [
        "CON",  # Reserved name
        "COM1",  # Reserved name
        "LPT1",  # Reserved name
        "-invalid",  # Starts with hyphen
        "_invalid",  # Starts with underscore
        "invalid@project",  # Invalid character
        "a" * 251,  # Too long (255 - len('.skore'))
    ]

    for name in invalid_names:
        with pytest.raises(ProjectCreationError):
            _create(name)


def test_create_project_existing_no_overwrite(tmp_path):
    """Test creating project with existing name without overwrite flag."""
    project_path = tmp_path / "existing_project"

    _create(project_path)
    with pytest.raises(FileExistsError):
        _create(project_path)


def test_create_project_permission_error(tmp_path, monkeypatch):
    """Test project creation with permission error."""

    def mock_mkdir(*args, **kwargs):
        raise PermissionError("Permission denied")

    # Patch Path.mkdir to raise PermissionError
    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    with pytest.raises(ProjectPermissionError):
        _create(tmp_path / "permission_denied")


def test_create_project_general_error(tmp_path, monkeypatch):
    """Test project creation with general error."""

    def mock_mkdir(*args, **kwargs):
        raise RuntimeError("Unknown error")

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    with pytest.raises(ProjectCreationError):
        _create(tmp_path / "error_project")


def test_create_project_subdirectory_errors(tmp_path, monkeypatch):
    """Test project creation fails when unable to create subdirectories."""

    original_mkdir = Path.mkdir
    failed_path = None

    def mock_mkdir(*args, **kwargs):
        nonlocal failed_path
        self = args[0] if args else kwargs.get("self")

        # Let the project directory creation succeed
        if self.name.endswith(".skore"):
            return original_mkdir(*args, **kwargs)

        # Fail if this is the path we want to fail
        if failed_path and self.name == failed_path:
            raise RuntimeError(f"Failed to create {failed_path} directory")

        return original_mkdir(*args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    # Test items directory creation failure
    failed_path = "items"
    with pytest.raises(ProjectCreationError) as exc_info:
        _create(tmp_path / "failed_items")
    assert "Unable to create project file" in str(exc_info.value)
    assert (tmp_path / "failed_items.skore").exists()

    # Test views directory creation failure
    failed_path = "views"
    with pytest.raises(ProjectCreationError) as exc_info:
        _create(tmp_path / "failed_views")
    assert "Unable to create project file" in str(exc_info.value)
    assert (tmp_path / "failed_views.skore").exists()

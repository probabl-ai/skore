import subprocess

import pytest
from skore.create_project import (
    ImproperProjectName,
    ProjectAlreadyExists,
    ProjectCreationError,
    ProjectNameTooLong,
    create_project,
    validate_project_name,
)

test_cases = [
    (
        "a" * 250,
        (False, ProjectNameTooLong()),
    ),
    (
        "%",
        (False, ImproperProjectName()),
    ),
    (
        "hello world",
        (False, ImproperProjectName()),
    ),
]


@pytest.mark.parametrize("project_name,expected", test_cases)
def test_validate_project_name(project_name, expected):
    result, exception = validate_project_name(project_name)
    expected_result, expected_exception = expected
    assert result == expected_result
    assert type(exception) is type(expected_exception)


@pytest.mark.parametrize("project_name", ["hello", "hello.skore"])
def test_create_project(project_name, tmp_path):
    create_project(project_name, working_dir=tmp_path)
    assert (tmp_path / "hello.skore").exists()


# TODO: If using fixtures in test cases is possible, join this with
# `test_create_project`
def test_create_project_absolute_path(tmp_path):
    create_project(tmp_path / "hello")
    assert (tmp_path / "hello.skore").exists()


def test_create_project_fails_if_file_exists(tmp_path):
    create_project(tmp_path / "hello")
    assert (tmp_path / "hello.skore").exists()
    with pytest.raises(ProjectAlreadyExists):
        create_project(tmp_path / "hello")


def test_create_project_fails_if_permission_denied(tmp_path):
    with pytest.raises(ProjectCreationError):
        create_project("/")


@pytest.mark.parametrize("project_name", ["hello.txt", "%%%", "COM1"])
def test_create_project_fails_if_invalid_name(project_name, tmp_path):
    with pytest.raises(ProjectCreationError):
        create_project(project_name, working_dir=tmp_path)


def test_create_project_cli_default_argument(tmp_path):
    completed_process = subprocess.run(
        f"python -m skore create --working-dir {tmp_path}".split(), capture_output=True
    )
    completed_process.check_returncode()
    assert (tmp_path / "project.skore").exists()


def test_create_project_cli_ends_in_skore(tmp_path):
    completed_process = subprocess.run(
        f"python -m skore create hello.skore --working-dir {tmp_path}".split(),
        capture_output=True,
    )
    completed_process.check_returncode()
    assert (tmp_path / "hello.skore").exists()


def test_create_project_cli_invalid_name(tmp_path):
    completed_process = subprocess.run(
        f"python -m skore create hello.txt --working-dir {tmp_path}".split(),
        capture_output=True,
    )
    with pytest.raises(subprocess.CalledProcessError):
        completed_process.check_returncode()
    assert b"ImproperProjectName" in completed_process.stderr

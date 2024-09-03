import subprocess

import pytest
from mandr.create_project import (
    ImproperProjectName,
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


@pytest.mark.parametrize("project_name", ["hello", "hello.mandr"])
def test_create_project(project_name, tmp_path):
    create_project(project_name, working_dir=tmp_path)
    assert (tmp_path / "hello.mandr").exists()


# TODO: If using fixtures in test cases is possible, join this with
# `test_create_project`
def test_create_project_absolute_path(tmp_path):
    create_project(tmp_path / "hello")
    assert (tmp_path / "hello.mandr").exists()


def test_create_project_fails_if_file_exists(tmp_path):
    create_project(tmp_path / "hello")
    assert (tmp_path / "hello.mandr").exists()
    with pytest.raises(ProjectCreationError):
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
        f"python -m mandr create --working-dir {tmp_path}".split(), capture_output=True
    )
    assert (
        f"Project file '{tmp_path}/project.mandr' was successfully created.".encode()
        in completed_process.stdout
    )
    assert (tmp_path / "project.mandr").exists()


def test_create_project_cli_ends_in_mandr(tmp_path):
    completed_process = subprocess.run(
        f"python -m mandr create hello.mandr --working-dir {tmp_path}".split(),
        capture_output=True,
    )
    assert (
        f"Project file '{tmp_path}/hello.mandr' was successfully created.".encode()
        in completed_process.stdout
    )
    assert (tmp_path / "hello.mandr").exists()


def test_create_project_cli_invalid_name(tmp_path):
    completed_process = subprocess.run(
        f"python -m mandr create hello.txt --working-dir {tmp_path}".split(),
        capture_output=True,
    )
    with pytest.raises(subprocess.CalledProcessError):
        completed_process.check_returncode()
    assert b"ImproperProjectName" in completed_process.stderr

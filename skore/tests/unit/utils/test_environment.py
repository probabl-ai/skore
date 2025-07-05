import sys
from unittest.mock import patch

import pytest
from skore._utils._environment import get_environment_info, is_environment_notebook_like


@pytest.fixture
def mock_sys_attributes():
    """Fixture to control sys attributes"""
    with (
        patch.object(sys, "ps1", create=True),
        patch.object(sys, "executable", "python3"),
        patch.object(sys, "version", "3.8.0"),
    ):
        yield


def test_get_environment_info_standard_python(mock_sys_attributes):
    """Test environment detection for standard Python execution"""
    with patch.dict("os.environ", {}, clear=True):
        info = get_environment_info()

        assert isinstance(info, dict)
        assert info["is_jupyter"] is False
        assert info["is_vscode"] is False
        assert info["is_interactive"] is True  # Due to mocked sys.ps1
        assert info["environment_name"] == "standard_python"
        assert info["details"]["python_executable"] == "python3"
        assert info["details"]["python_version"] == "3.8.0"


def test_get_environment_info_vscode():
    """Test environment detection for VS Code"""
    with patch.dict("os.environ", {"VSCODE_PID": "12345"}):
        info = get_environment_info()

        assert info["is_vscode"] is True
        assert "vscode" in info["environment_name"]


@patch.dict(
    "os.environ", {}, clear=True
)  # to avoid false vscode detection when running tests from vscode test runner
@patch("skore._utils._environment.get_ipython", create=True)
def test_get_environment_info_jupyter(mock_get_ipython):
    """Test environment detection for Jupyter"""
    mock_get_ipython.return_value.__class__.__name__ = "ZMQInteractiveShell"

    info = get_environment_info()

    assert info["is_jupyter"] is True
    assert info["environment_name"] == "jupyter"
    assert info["details"]["ipython_shell"] == "ZMQInteractiveShell"


def test_is_environment_notebook_like():
    """Test notebook-like environment detection"""
    with patch.dict("os.environ", {"VSCODE_PID": "12345"}):
        assert is_environment_notebook_like() is True

    with patch.dict("os.environ", {}, clear=True):
        assert is_environment_notebook_like() is False


@patch("skore._utils._environment.get_ipython", create=True)
def test_is_environment_notebook_like_jupyter(mock_get_ipython):
    """Test notebook-like environment detection for Jupyter"""
    mock_get_ipython.return_value.__class__.__name__ = "ZMQInteractiveShell"

    assert is_environment_notebook_like() is True


@patch.dict(
    "os.environ", {}, clear=True
)  # to avoid false vscode detection when running tests from vscode test runner
@patch("skore._utils._environment.get_ipython", create=True)
def test_get_environment_info_ipython_terminal(mock_get_ipython):
    """Test environment detection for IPython terminal"""
    mock_get_ipython.return_value.__class__.__name__ = "TerminalInteractiveShell"

    info = get_environment_info()

    assert info["is_jupyter"] is False
    assert info["environment_name"] == "ipython_terminal"
    assert info["details"]["ipython_shell"] == "TerminalInteractiveShell"


@patch("skore._utils._environment.os.environ", {"VSCODE_PID": "12345"})
@patch("skore._utils._environment.sys")
def test_get_environment_info_vscode_interactive(mock_sys):
    """Test environment detection for VSCode in interactive mode"""
    mock_sys.ps1 = True

    info = get_environment_info()

    assert info["is_vscode"] is True
    assert info["is_interactive"] is True
    assert info["environment_name"] == "vscode_interactive"

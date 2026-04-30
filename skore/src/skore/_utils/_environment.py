import os
import sys
from typing import Any


def get_environment_info() -> dict[str, Any]:
    """Detect the current Python execution environment.

    Returns a dictionary with information about the environment.
    """
    env_info: dict[str, Any] = {
        "is_jupyter": False,
        "is_vscode": False,
        "is_interactive": False,
        "is_sphinx_build": False,
        "environment_name": "standard_python",
        "details": {},
    }

    env_info["is_interactive"] = hasattr(sys, "ps1")

    try:
        # get_ipython() is defined when running in Jupyter or IPython
        # there is no need to import IPython here
        ipython = get_ipython()  # type: ignore[name-defined]
    except NameError:
        pass
    else:
        shell = ipython.__class__.__name__

        env_info["details"]["ipython_shell"] = shell

        # Jupyter notebook/lab or Jupyterlite
        if (shell == "ZMQInteractiveShell") or ("pyodide" in str(ipython.__class__)):
            env_info["is_jupyter"] = True
            env_info["environment_name"] = "jupyter"
        # IPython terminal
        elif shell == "TerminalInteractiveShell":
            env_info["environment_name"] = "ipython_terminal"

    if "VSCODE_PID" in os.environ:
        env_info["is_vscode"] = True
        if env_info["is_interactive"]:
            env_info["environment_name"] = "vscode_interactive"
        else:
            env_info["environment_name"] = "vscode_script"

    if "SPHINX_BUILD" in os.environ:
        env_info["is_sphinx_build"] = True

    env_info["details"]["python_executable"] = sys.executable
    env_info["details"]["python_version"] = sys.version

    return env_info


def is_environment_notebook_like() -> bool:
    """Return `True` if the execution context can render HTML. `False` otherwise."""
    info = get_environment_info()
    return info["is_vscode"] or info["is_jupyter"]


def is_environment_sphinx_build() -> bool:
    """Return `True` if the execution context is a sphinx build. `False` otherwise."""
    info = get_environment_info()
    return info["is_sphinx_build"]

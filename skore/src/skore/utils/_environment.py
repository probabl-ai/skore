import os
import sys


def get_environment_info():
    """Detect the current Python execution environment.

    Returns a dictionary with information about the environment.
    """
    env_info = {
        "is_jupyter": False,
        "is_vscode": False,
        "is_interactive": False,
        "environment_name": "standard_python",
        "details": {},
    }

    # Check for interactive mode
    env_info["is_interactive"] = hasattr(sys, "ps1")

    # Check for Jupyter
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        env_info["details"]["ipython_shell"] = shell

        if shell == "ZMQInteractiveShell":  # Jupyter notebook/lab
            env_info["is_jupyter"] = True
            env_info["environment_name"] = "jupyter"
        elif shell == "TerminalInteractiveShell":  # IPython terminal
            env_info["environment_name"] = "ipython_terminal"
    except NameError:
        pass

    # Check for VSCode
    if "VSCODE_PID" in os.environ:
        env_info["is_vscode"] = True
        if env_info["is_interactive"]:
            env_info["environment_name"] = "vscode_interactive"
        else:
            env_info["environment_name"] = "vscode_script"

    # Add additional environment details
    env_info["details"]["python_executable"] = sys.executable
    env_info["details"]["python_version"] = sys.version

    return env_info


def is_environment_html_capable() -> bool:
    """Return `True` if the execution context dcan render HTML. `False` otherwise."""
    info = get_environment_info()
    return info["is_vscode"] or info["is_jupyter"]

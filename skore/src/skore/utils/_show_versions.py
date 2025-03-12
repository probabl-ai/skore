"""
Utility methods to print system info for debugging.

adapted from :func:`sklearn.show_versions`
"""

import importlib
import platform
import re
import sys
from typing import Any, Union


def _get_sys_info() -> dict[str, Any]:
    """System information.

    Returns
    -------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info() -> dict[str, Any]:
    """Overview of the installed version of main dependencies.

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """
    from importlib.metadata import PackageNotFoundError, version

    deps = ["pip"]

    raw_requirements = importlib.metadata.requires("skore")
    requirements: list[str] = [] if raw_requirements is None else raw_requirements

    for requirement in filter(lambda r: "; extra" not in r, requirements):
        # Extract just the package name before any version specifiers
        package_name: str = re.split(r"[<>=~!]", requirement)[0].strip()
        deps.append(package_name)

    deps_info: dict[str, Union[str, None]] = {
        "skore": version("skore"),
    }

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions() -> None:
    """Print useful debugging information.

    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> from skore import show_versions
    >>> show_versions()
    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")  # noqa: T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T201

    print("\nPython dependencies:")  # noqa: T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T201

"""
Utility methods to print system info for debugging.

adapted from :func:`sklearn.show_versions`
"""

import importlib
import platform
import sys


def _get_sys_info():
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


def _get_deps_info():
    """Overview of the installed version of main dependencies.

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries

    """
    from importlib.metadata import PackageNotFoundError, version

    deps = ["pip", "setuptools"]

    requirements = importlib.metadata.requires("skore")
    for requirement in filter(lambda r: "; extra" not in r, requirements):
        deps.append(requirement)

    deps_info = {
        "skore": version("skore"),
    }

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions():
    """Print useful debugging information.

    Examples
    --------
    >>> from skore import show_versions
    >>> show_versions()  # doctest: +SKIP
    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")  # noqa: T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T201

    print("\nPython dependencies:")  # noqa: T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T201

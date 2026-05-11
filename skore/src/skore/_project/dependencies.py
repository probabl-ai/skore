"""Tools used to assert that optional dependencies are installed."""

import re
from importlib.metadata import PackageNotFoundError, requires, version


def optional_dependencies(extra: str) -> list[str]:
    """List optional dependencies required by ``skore[extra]``."""
    dependencies = requires("skore")
    regex = re.compile(rf"extra\s*==\s*['\"]{extra}['\"]")
    names = []

    assert dependencies is not None, "No `skore` dependencies found"

    for dependency in dependencies:
        if regex.search(dependency):
            head = dependency.split(";", 1)[0].strip()
            name = re.split(r"\s*(?:<|>|=|!|~|\[)", head, maxsplit=1)[0].strip()

            names.append(name)

    return names


def assert_optional_dependencies_installed(extra: str) -> None:
    """Assert if every optional dependency required by ``skore[extra]`` is installed."""
    for dependency in optional_dependencies(extra):
        try:
            version(dependency)
        except PackageNotFoundError:
            raise ImportError(
                f"Missing `{dependency}` library: please install `skore[{extra}]`."
            ) from None

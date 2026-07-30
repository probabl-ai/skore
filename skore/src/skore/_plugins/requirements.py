"""Infer installed package requirements from currently imported modules."""

import importlib.metadata
import logging
import pathlib
import platform
import site
import sys
import sysconfig
import types
import typing

logger = logging.getLogger(__name__)


def is_local_module(module: types.ModuleType) -> bool:
    """
    Return whether ``module`` was loaded from outside site-packages.

    True for editable installs, source trees, and other paths that are neither the
    standard library nor an installed distribution in site-packages.
    """
    sitepackages = [pathlib.Path(path).resolve() for path in site.getsitepackages()]

    if site.ENABLE_USER_SITE and site.getusersitepackages():
        sitepackages.append(pathlib.Path(site.getusersitepackages()).resolve())

    origin = (module.__spec__ and module.__spec__.origin) or getattr(
        module, "__file__", None
    )

    if (not origin) or (origin in {"built-in", "frozen"}):
        return False

    path = pathlib.Path(origin).resolve()
    stdlib = pathlib.Path(sysconfig.get_path("stdlib")).resolve()

    if path.is_relative_to(stdlib):
        return False

    return not any(map(path.is_relative_to, sitepackages))


class Requirement(typing.TypedDict):
    """A distribution name and its installed version."""

    name: str
    version: str | None


class Requirements(typing.TypedDict):
    """Python version and inferred distribution requirements."""

    python: str
    requirements: list[Requirement]


def infer() -> Requirements:
    """
    Infer distribution requirements from modules currently in ``sys.modules``.

    Maps each imported top-level package to its distribution via
    :func:`importlib.metadata.packages_distributions`, then records the installed
    version. Local or editable packages outside site-packages are skipped and a
    warning is emitted once per package name.
    """
    module_to_requirement = importlib.metadata.packages_distributions()
    requirement_to_version = {}
    warned = set()

    # Snapshot: importlib.metadata / console I/O can load modules and mutate
    # sys.modules while we iterate.
    for module in list(sys.modules.values()):
        if module is None:
            continue

        name = (module.__spec__ and module.__spec__.name) or module.__name__
        top_level_name = name.partition(".")[0]

        if top_level_name in sys.stdlib_module_names:
            continue

        if requirements := module_to_requirement.get(top_level_name):
            for requirement in requirements:
                if requirement not in requirement_to_version:
                    requirement_to_version[requirement] = importlib.metadata.version(
                        requirement
                    )
        else:
            # No distribution mapping: Cython/C-extension aliases, runtime entrypoints
            # (__main__), or local/dev imports outside site-packages.
            logging.debug(module)

            if is_local_module(module) and (top_level_name not in warned):
                warned.add(top_level_name)
                logger.warning(
                    "\033[38;5;208m"
                    f"Package \033[1;3m{top_level_name}\033[22;23m seems to be an "
                    "editable or local install (loaded from outside site-packages). "
                    "It will not be recorded in the inferred requirements."
                    "\033[0m"
                )

    return Requirements(
        python=platform.python_version(),
        requirements=[
            Requirement(
                name=name,
                version=version,
            )
            for name, version in sorted(requirement_to_version.items())
        ],
    )

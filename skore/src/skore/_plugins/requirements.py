"""Infer installed package requirements from currently imported modules."""

import functools
import importlib.metadata
import logging
import pathlib
import sys
import sysconfig
import types
import typing
import warnings

import packaging.utils

MODULE_TO_REQUIREMENT = importlib.metadata.packages_distributions()
version = functools.cache(importlib.metadata.version)
logger = logging.getLogger(__name__)


def is_local_module(module: types.ModuleType) -> bool:
    """
    Return whether ``module`` was loaded from outside the standard library.

    True for editable installs, source trees, and other non-stdlib paths.
    False when there is no usable file origin (``None``, ``built-in``,
    ``frozen``) or the origin is under the stdlib path.
    """
    origin = (module.__spec__ and module.__spec__.origin) or getattr(
        module, "__file__", None
    )

    if (origin is None) or (origin in {"built-in", "frozen"}):
        return False

    path = pathlib.Path(origin).resolve()
    stdlib = pathlib.Path(sysconfig.get_path("stdlib")).resolve()

    return not path.is_relative_to(stdlib)


class Requirement(typing.TypedDict):
    """A distribution name and its installed version."""

    name: str
    version: str


def infer() -> list[Requirement]:
    """
    Infer distribution requirements from modules currently in ``sys.modules``.

    Maps each imported top-level package to its distribution via
    :func:`importlib.metadata.packages_distributions`, then records the installed
    version. Unmapped packages loaded from outside the standard library are
    skipped and a warning is emitted once per package name.
    """
    requirement_to_version = {}
    warned = set()

    # Snapshot: importlib.metadata/console I/O can load modules and mutate sys.modules
    # while we iterate.
    for module in list(sys.modules.values()):
        # sys.modules values are usually modules, but can be None (cleared/failed
        # imports) or non-modules (legacy typing.io/re classes) that lack __spec__.
        if not isinstance(module, types.ModuleType):
            continue

        name = (module.__spec__ and module.__spec__.name) or module.__name__
        top_level_name = name.partition(".")[0]

        if top_level_name in sys.stdlib_module_names:
            continue

        if requirements := MODULE_TO_REQUIREMENT.get(top_level_name):
            for requirement in requirements:
                if requirement not in requirement_to_version:
                    requirement_to_version[requirement] = version(requirement)
        else:
            # No distribution mapping: Cython/C-extension aliases, runtime entrypoints
            # (__main__), or local/dev imports outside site-packages.
            logger.debug(module)

            if is_local_module(module) and (top_level_name not in warned):
                warned.add(top_level_name)
                warnings.warn(
                    (
                        f"Package {top_level_name} seems to be an editable or local "
                        "install (loaded from outside site-packages). It will not be "
                        "recorded in the inferred requirements."
                    ),
                    stacklevel=2,
                )

    return [
        Requirement(
            name=packaging.utils.canonicalize_name(name),
            version=packaging.utils.canonicalize_version(
                version,
                strip_trailing_zero=False,
            ),
        )
        for name, version in sorted(requirement_to_version.items())
    ]

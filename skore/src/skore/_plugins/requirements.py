"""Infer installed package requirements from currently imported modules."""

import functools
import importlib.metadata
import itertools
import logging
import operator
import pathlib
import site
import sys
import sysconfig
import types
import typing
import warnings

import packaging.utils

logger = logging.getLogger(__name__)

version = functools.cache(importlib.metadata.version)
MODULE_TO_REQUIREMENT = importlib.metadata.packages_distributions()
SITE_PACKAGES_DIRPATHS = tuple(
    {
        pathlib.Path(path).resolve()
        for path in itertools.chain(
            map(sysconfig.get_path, ("stdlib", "purelib", "platlib")),
            site.getsitepackages(),
            (
                [site.getusersitepackages()]
                if site.ENABLE_USER_SITE and site.getusersitepackages()
                else []
            ),
        )
    }
)


def is_editable_or_local_module(module: types.ModuleType) -> bool:
    """
    Return whether ``module`` was loaded from outside known install locations.

    True for editable installs, source trees, and other paths outside the
    standard library and site-packages. False when there is no usable file
    origin (``None``, ``built-in``, ``frozen``) or the origin is under a
    site-packages path.
    """
    origin = (module.__spec__ and module.__spec__.origin) or getattr(
        module, "__file__", None
    )

    if (origin is None) or (origin in {"built-in", "frozen"}):
        return False

    path = pathlib.Path(origin).resolve()

    return not any(map(path.is_relative_to, SITE_PACKAGES_DIRPATHS))


class Requirement(typing.TypedDict):
    """A distribution name and its installed version."""

    name: str
    version: str


def infer() -> list[Requirement]:
    """
    Infer distribution requirements from modules currently in ``sys.modules``.

    Maps each imported top-level package to its distribution via
    :func:`importlib.metadata.packages_distributions`, then records the installed
    version. Packages loaded from outside known install locations (editable or
    local installs) are skipped and a warning is emitted once per package name,
    even when a distribution mapping exists.
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

        if is_editable_or_local_module(module):
            if top_level_name not in warned:
                warned.add(top_level_name)
                warnings.warn(
                    (
                        f"Package {top_level_name} seems to be an editable or local "
                        "install (loaded from outside site-packages). It will not be "
                        "recorded in the inferred requirements."
                    ),
                    stacklevel=2,
                )
            continue

        if requirements := MODULE_TO_REQUIREMENT.get(top_level_name):
            for requirement in requirements:
                if requirement not in requirement_to_version:
                    requirement_to_version[requirement] = version(requirement)
        else:
            # No distribution mapping: Cython/C-extension aliases, runtime entrypoints
            # (__main__), etc.
            logger.debug(module)

    return sorted(
        (
            Requirement(
                name=packaging.utils.canonicalize_name(name),
                version=packaging.utils.canonicalize_version(
                    version,
                    strip_trailing_zero=False,
                ),
            )
            for name, version in requirement_to_version.items()
        ),
        key=operator.itemgetter("name"),
    )

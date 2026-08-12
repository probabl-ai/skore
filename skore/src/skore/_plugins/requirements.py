"""Infer installed package requirements from currently imported modules.

Results depend entirely on ``sys.modules`` at call time: two successive calls
can return different requirements if imports (or unloadings) happen in between.
"""

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
MODULE_TO_DISTRIBUTIONS = importlib.metadata.packages_distributions()
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


@functools.cache
def __distribution_files(name: str) -> frozenset[pathlib.Path]:
    files = importlib.metadata.distribution(name).files or []
    located = set()

    for file in files:
        try:
            located.add(pathlib.Path(file.locate()).resolve())
        except OSError:
            continue

    return frozenset(located)


def __distribution(top_level_name: str, module: types.ModuleType) -> str | None:
    """
    Return the distribution name that owns ``module``.

    Uses :func:`importlib.metadata.packages_distributions` for the top-level
    name. When several distributions share that name (namespace packages), only
    the distribution whose installed files include the module origin is returned.
    A bare namespace root with no file origin owns nothing on its own.
    """
    candidates = MODULE_TO_DISTRIBUTIONS.get(top_level_name)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # When several distributions share that top-level name (namespace packages),
    # we have to inspect each distribution and retrieve the one that contains the module
    # file.

    origin = (module.__spec__ and module.__spec__.origin) or getattr(
        module, "__file__", None
    )

    if (origin is None) or (origin in {"built-in", "frozen"}):
        return None

    origin = pathlib.Path(origin).resolve()

    return next(
        (
            candidate
            for candidate in candidates
            if origin in __distribution_files(candidate)
        ),
        None,
    )


def infer() -> list[Requirement]:
    """
    Infer distribution requirements from modules currently in ``sys.modules``.

    The returned list depends entirely on the contents of ``sys.modules`` at the
    moment of the call. Two successive calls can therefore differ: any import (or
    module removal) between them changes which distributions are discovered.

    Maps each imported module to its distribution via
    :func:`importlib.metadata.packages_distributions`. For namespace packages,
    only the distribution that owns the imported file is recorded. Packages
    loaded from outside known install locations (editable or local installs) are
    skipped and a warning is emitted once per package name, even when a
    distribution mapping exists.
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

        if distribution := __distribution(top_level_name, module):
            if distribution not in requirement_to_version:
                requirement_to_version[distribution] = version(distribution)
        else:
            # No distribution mapping: Cython/C-extension aliases, runtime entrypoints
            # (__main__), bare namespace roots, etc.
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

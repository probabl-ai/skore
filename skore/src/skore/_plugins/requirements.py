import importlib.metadata
import pathlib
import platform
import site
import sys
import sysconfig
import typing


def _site_dirs() -> list[pathlib.Path]:
    dirs = [pathlib.Path(p).resolve() for p in site.getsitepackages()]

    if site.ENABLE_USER_SITE and site.getusersitepackages():
        dirs.append(pathlib.Path(site.getusersitepackages()).resolve())

    return dirs


def is_local_module(module) -> bool:
    origin = getattr(module, "__file__", None) or getattr(
        getattr(module, "__spec__", None), "origin", None
    )

    if not origin or origin in {"built-in", "frozen"}:
        return False

    path = pathlib.Path(origin).resolve()
    stdlib = pathlib.Path(sysconfig.get_path("stdlib")).resolve()

    if path.is_relative_to(stdlib):
        return False

    return not any(path.is_relative_to(d) for d in _site_dirs())


class Requirement(typing.TypedDict):
    name: str
    version: str | None


class Requirements(typing.TypedDict):
    python: str
    requirements: list[Requirement]


def infer() -> Requirements:
    module_to_requirement = importlib.metadata.packages_distributions()
    requirement_to_version = {}
    nonlocalm = []
    localm = []

    for module in sys.modules.values():
        name = ((spec := module.__spec__) and spec.name) or module.__name__
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
            # Cython internals
            # C extensions
            # Runtime entrypoints
            # Editable packages: origin outside site-packages (repo, cwd, editable .pth path)

            if not is_local_module(module):
                nonlocalm.append((module, top_level_name))
            else:
                localm.append((module, top_level_name))

    return (
        nonlocalm,
        localm,
        Requirements(
            python=platform.python_version(),
            requirements=[
                Requirement(
                    name=name,
                    version=version,
                )
                for name, version in sorted(requirement_to_version.items())
            ],
        ),
    )

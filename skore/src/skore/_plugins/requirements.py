import importlib.metadata
import platform
import sys
import typing


class Requirement(typing.TypedDict):
    name: str
    version: str | None


class Requirements(typing.TypedDict):
    python: str
    requirements: list[Requirement]


def version(requirement: str) -> str | None:
    try:
        return importlib.metadata.version(requirement)
    except importlib.metadata.PackageNotFoundError:
        return None


def infer() -> Requirements:
    module_to_requirement = importlib.metadata.packages_distributions()
    requirement_to_version = {}

    for name, module in sys.modules.items():
        if module is None:
            continue

        top_level_module = name.partition(".")[0]

        if top_level_module in sys.stdlib_module_names:
            continue

        if requirements := module_to_requirement.get(top_level_module):
            for requirement in requirements:
                if requirement not in requirement_to_version:
                    requirement_to_version[requirement] = version(requirement)
        else:
            ...

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

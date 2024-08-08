"""
    [storage]
    type = "filesystem"

    [storage.filesystem]
    directory = "/tmp/mandr"

Il faut pouvoir récupérer la configuration depuis :

- /p/r/o/j/e/c/t/mandr.toml -> [storage] / [storage.filesystem]
- /p/r/o/j/e/c/t/pyproject.toml -> [tool.mandr.storage] / [tool.mandr.storage.filesystem]

À terme, il faudra pouvoir récupérer depuis :

- Linux: $XDG_CONFIG_HOME/pymandr or ~/.config/pymandr
- Windows: %APPDATA%\pymandr
- MacOS: ~/Library/Application Support/pymandr

Il faudra écrire un README *exhaustif*.

Notes
-----
Comment récupérer un fichier du package client ?

Sources
-------

Fortement inspiré de poetry :

- https://python-poetry.org/docs/pyproject/
- https://python-poetry.org/docs/configuration
- https://github.com/python-poetry/poetry/blob/b6bad9c73c9e7e74a1a10df22210837a1e231ba1/src/poetry/locations.py
- https://github.com/python-poetry/poetry/blob/b6bad9c73c9e7e74a1a10df22210837a1e231ba1/src/poetry/config/config.py
"""

import copy
import pathlib

import tomllib

__configuration = None
__configuration_filepath = None
__DEFAULT_CONFIGURATION = {
    "storage": {
        "type": "filesystem",
        "filesystem": {
            "directory": (pathlib.Path.cwd() / ".mandr"),
        },
    },
}


class TOMLFile:
    def __init__(self, filepath: pathlib.Path, /, pyproject: bool = False):
        self.__filepath = filepath
        self.__pyproject = pyproject

    def exists(self) -> bool:
        return self.__filepath.exists()

    def read(self) -> dict:
        with open(self.__filepath, "rb") as file:
            configuration = tomllib.load(file)

        if self.__pyproject:
            try:
                return configuration["tool"]["mandr"]
            except KeyError:
                return {}

        return configuration


def merge(d1: dict, d2: dict):
    for key in d1.keys() & d2.keys():
        if isinstance(d1[key], dict):
            if isinstance(d2[key], dict):
                merge(d1[key], d2[key])
        else:
            d1[key] = d2[key]


# class Configuration:
#     def __init__(filepath: Path, configuration):
#         self.__filepath = filepath
#         self.__configuration = configuration


def configure(filepath: Path, *, reload: bool = False):
    if not filepath.exists():
        raise FileNotFoundError




def Configuration(*, reload: bool = False) -> dict:
    global __configuration
    global __DEFAULT_CONFIGURATION

    if (__configuration is None) or reload:
        default = copy.deepcopy(__DEFAULT_CONFIGURATION)

        for file in [
            TOMLFile(pathlib.Path("~/.config/mandr.toml")),
            TOMLFile(pathlib.Path("~/mandr.toml")),
            TOMLFile(pathlib.Path("./mandr.toml")),
            TOMLFile(pathlib.Path("./pyproject.toml"), pyproject=True),
        ]:
            if file.exists():
                merge(default, file.read())

        __configuration = default

    return copy.deepcopy(__configuration)


def foo(): ...


import inspect
import pathlib

stack = inspect.stack()
packages = {}

for frameinfo in stack[1:]:
    filepath = pathlib.Path(frameinfo.filename)

    if filepath.exists():
        module = inspect.getmodule(frameinfo.frame)
        package = module.__name__.split(".", 1)[0]

        package.append(package)

breakpoint()

# module.__path__
# https://docs.python.org/3/library/importlib.metadata.html
# from importlib import metadata
breakpoint()

parent_frameinfo = stack[1]
parent_filename = parent_frameinfo.filename
parent_frame = parent_frameinfo.frame


parent_module = inspect.getmodule(parent_frame)
parent_package = parent_module.__package__

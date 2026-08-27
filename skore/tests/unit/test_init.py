from importlib.metadata import version

from skore import __version__


def test__version__returns_version():
    assert __version__ == version("skore")

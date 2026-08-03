from importlib.machinery import ModuleSpec
from types import ModuleType

from skore._plugins.requirements import is_local_module


class Module(ModuleType):
    def __init__(self, name: str, origin: str | None):
        super().__init__(name)

        self.__file__ = origin
        self.__spec__ = ModuleSpec(name, loader=None, origin=origin)


class TestIsLocalModule:
    def test_no_origin(self, tmp_path):
        assert not is_local_module(Module("_cython_3_2_4", None))

    def test_stdlib(self):
        import pathlib
        import sysconfig

        assert not is_local_module(
            Module(
                "_sysconfigdata",
                (pathlib.Path(sysconfig.get_path("stdlib")) / "_sysconfigdata.py"),
            )
        )

    def test_editable_install(self, tmp_path):
        assert is_local_module(
            Module(
                "pkg",
                (tmp_path / "pkg" / "src" / "pkg" / "__init__.py"),
            )
        )


class TestInfer: ...

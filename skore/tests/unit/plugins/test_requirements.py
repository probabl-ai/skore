from importlib.machinery import ModuleSpec
from types import ModuleType

from pytest import warns

from skore._plugins import requirements


class Module(ModuleType):
    def __init__(self, name: str, origin: str | None):
        super().__init__(name)

        self.__file__ = origin
        self.__spec__ = ModuleSpec(name, loader=None, origin=origin)


class TestIsLocalModule:
    def test_no_origin(self, tmp_path):
        assert not requirements.is_local_module(Module("_cython_3_2_4", None))

    def test_stdlib(self):
        import pathlib
        import sysconfig

        assert not requirements.is_local_module(
            Module(
                "_sysconfigdata",
                (pathlib.Path(sysconfig.get_path("stdlib")) / "_sysconfigdata.py"),
            )
        )

    def test_editable_install(self, tmp_path):
        assert requirements.is_local_module(
            Module(
                "pkg",
                (tmp_path / "pkg" / "src" / "pkg" / "__init__.py"),
            )
        )


class TestInfer:
    def test_records_distributions_once(self, monkeypatch):
        import numpy
        import numpy.linalg

        monkeypatch.setattr(
            requirements.sys,
            "modules",
            {
                "numpy": numpy,
                "numpy.linalg": numpy.linalg,
            },
        )

        assert requirements.infer() == [{"name": "numpy", "version": numpy.__version__}]

    def test_records_aliased_distributions(self, monkeypatch):
        import numpy
        import numpy.linalg
        import sklearn
        import sklearn.base

        monkeypatch.setattr(
            requirements.sys,
            "modules",
            {
                "numpy": numpy,
                "numpy.linalg": numpy.linalg,
                "sklearn": sklearn,
                "sklearn.base": sklearn.base,
            },
        )

        assert requirements.infer() == [
            {"name": "numpy", "version": numpy.__version__},
            {"name": "scikit-learn", "version": sklearn.__version__},
        ]

    def test_skips_stdlib_modules(self, tmp_path, monkeypatch):
        import json

        import numpy
        import numpy.linalg
        import sklearn
        import sklearn.base

        monkeypatch.setattr(
            requirements.sys,
            "modules",
            {
                "json": json,
                "numpy": numpy,
                "numpy.linalg": numpy.linalg,
                "sklearn": sklearn,
                "sklearn.base": sklearn.base,
            },
        )

        assert requirements.infer() == [
            {"name": "numpy", "version": numpy.__version__},
            {"name": "scikit-learn", "version": sklearn.__version__},
        ]

    def test_skips_none_entries(self, monkeypatch):
        import json

        import numpy
        import numpy.linalg
        import sklearn
        import sklearn.base

        monkeypatch.setattr(
            requirements.sys,
            "modules",
            {
                "broken": None,
                "json": json,
                "numpy": numpy,
                "numpy.linalg": numpy.linalg,
                "sklearn": sklearn,
                "sklearn.base": sklearn.base,
            },
        )

        assert requirements.infer() == [
            {"name": "numpy", "version": numpy.__version__},
            {"name": "scikit-learn", "version": sklearn.__version__},
        ]

    def test_skips_non_module_entries(self, monkeypatch):
        import json

        import numpy
        import numpy.linalg
        import sklearn
        import sklearn.base

        class FakeTypingIo:
            """Legacy typing.io/re were classes registered in sys.modules."""

            __name__ = "io"

        monkeypatch.setattr(
            requirements.sys,
            "modules",
            {
                "typing.io": FakeTypingIo,
                "json": json,
                "numpy": numpy,
                "numpy.linalg": numpy.linalg,
                "sklearn": sklearn,
                "sklearn.base": sklearn.base,
            },
        )

        assert requirements.infer() == [
            {"name": "numpy", "version": numpy.__version__},
            {"name": "scikit-learn", "version": sklearn.__version__},
        ]

    def test_warns_once_for_editable_package(self, tmp_path, monkeypatch):
        origin = tmp_path / "pkg" / "src" / "pkg" / "__init__.py"

        import json
        import warnings

        import numpy
        import numpy.linalg
        import sklearn
        import sklearn.base

        monkeypatch.setattr(
            requirements.sys,
            "modules",
            {
                "broken": None,
                "json": json,
                "numpy": numpy,
                "numpy.linalg": numpy.linalg,
                "sklearn": sklearn,
                "sklearn.base": sklearn.base,
                "pkg": Module("pkg", origin),
                "pkg.utils": Module("pkg.utils", origin),
                "warnings": warnings,
            },
        )

        with warns(UserWarning, match=r"pkg seems to be an editable"):
            assert requirements.infer() == [
                {"name": "numpy", "version": numpy.__version__},
                {"name": "scikit-learn", "version": sklearn.__version__},
            ]

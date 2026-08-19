from importlib.machinery import ModuleSpec
from types import ModuleType

from pytest import warns

from skore.plugins import requirements


class Module(ModuleType):
    def __init__(self, name: str, origin: str | None):
        super().__init__(name)

        self.__file__ = origin
        self.__spec__ = ModuleSpec(name, loader=None, origin=origin)


class TestIsEditableOrLocalModule:
    def test_no_origin(self, tmp_path):
        assert not requirements.is_editable_or_local_module(
            Module("_cython_3_2_4", None)
        )

    def test_stdlib(self):
        import pathlib
        import sysconfig

        assert not requirements.is_editable_or_local_module(
            Module(
                "_sysconfigdata",
                (pathlib.Path(sysconfig.get_path("stdlib")) / "_sysconfigdata.py"),
            )
        )

    def test_site_packages(self):
        import numpy

        assert not requirements.is_editable_or_local_module(numpy)

    def test_editable_install(self, tmp_path):
        assert requirements.is_editable_or_local_module(
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

    def test_warns_fast_for_editable_package(self, tmp_path, monkeypatch):
        """
        Non-regression test for packages that are installed with setuptools in editable
        mode, where distributions are still mapped.

        Ensure `is_editable_or_local_module` is called before and routes well.
        """
        origin = tmp_path / "mypkg" / "src" / "mypkg" / "__init__.py"

        import warnings

        monkeypatch.setattr(requirements, "MODULE_TO_DISTRIBUTIONS", None)  # dict
        monkeypatch.setattr(requirements, "version", None)  # function
        monkeypatch.setattr(
            requirements.sys,
            "modules",
            {
                "mypkg": Module("mypkg", origin),
                "mypkg.utils": Module("mypkg.utils", origin),
                "warnings": warnings,
            },
        )

        with warns(UserWarning, match=r"mypkg seems to be an editable"):
            assert not requirements.infer()

    def test_records_only_owning_namespace_distribution(self, monkeypatch):
        """Namespace packages map one top-level name to several distributions."""
        import pathlib
        import sysconfig

        site_packages = pathlib.Path(sysconfig.get_path("purelib"))
        origin = site_packages / "google" / "protobuf" / "__init__.py"

        monkeypatch.setattr(
            requirements,
            "MODULE_TO_DISTRIBUTIONS",
            {"google": ["protobuf", "google-auth"]},
        )
        monkeypatch.setattr(
            requirements,
            "__distribution_files",
            lambda name: {
                "protobuf": frozenset({origin.resolve()}),
                "google-auth": frozenset(
                    {(site_packages / "google" / "auth" / "__init__.py").resolve()}
                ),
            }[name],
        )
        monkeypatch.setattr(
            requirements,
            "version",
            lambda name: {"protobuf": "5.0.0", "google-auth": "2.0.0"}[name],
        )
        monkeypatch.setattr(
            requirements.sys,
            "modules",
            {
                "google": Module("google", None),
                "google.protobuf": Module("google.protobuf", origin),
            },
        )

        assert requirements.infer() == [{"name": "protobuf", "version": "5.0.0"}]

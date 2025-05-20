from sys import version_info

from pytest import fixture, raises
from skore import Project

if version_info < (3, 10):
    from importlib_metadata import EntryPoint, EntryPoints
else:
    from importlib.metadata import EntryPoint, EntryPoints


class FakeLocalProject:
    def __init__(self, *args, **kwargs): ...


FakeHubProject = FakeLocalProject


class TestProject:
    @fixture(autouse=True)
    def monkeypatch_entrypoints(self, monkeypatch, request):
        entrypoints = EntryPoints(
            [
                EntryPoint(
                    name="local",
                    value=f"{__name__}:FakeLocalProject",
                    group="skore.plugins.project",
                ),
                EntryPoint(
                    name="hub",
                    value=f"{__name__}:FakeHubProject",
                    group="skore.plugins.project",
                ),
            ]
        )

        monkeypatch.setattr(
            "skore.project.project.entry_points", lambda **kwargs: entrypoints
        )

    def test_init_local(self):
        project = Project("<name>")

        assert isinstance(project, Project)
        assert isinstance(project._Project__project, FakeLocalProject)
        assert project._Project__mode == "local"
        assert project._Project__name == "<name>"

    def test_init_hub(self):
        project = Project("hub://<tenant>/<name>")

        assert isinstance(project, Project)
        assert isinstance(project._Project__project, FakeHubProject)
        assert project._Project__mode == "hub"
        assert project._Project__name == "<name>"

    def test_init_exception_no_plugin(self, monkeypatch, tmp_path):
        monkeypatch.undo()
        monkeypatch.setattr(
            "skore.project.project.entry_points", lambda **kwargs: EntryPoints()
        )

        with raises(SystemError, match="No project plugin found"):
            Project("<project>")

    def test_init_exception_unknown_plugin(self, monkeypatch, tmp_path):
        monkeypatch.undo()
        monkeypatch.setattr(
            "skore.project.project.entry_points",
            lambda **kwargs: EntryPoints(
                [
                    EntryPoint(
                        "hub",
                        f"{__file__}.FakeHubProject",
                        "skore.plugins.project",
                    )
                ]
            ),
        )

        with raises(
            ValueError,
            match=(
                "Unknown mode `local`. "
                "Please install the `skore-local-project` python package."
            ),
        ):
            Project("<name>")

    def test_mode(self):
        assert Project("<name>").mode == "local"
        assert Project("hub://<tenant>/<name>").mode == "hub"

    def test_name(self):
        assert Project("<name>").name == "<name>"
        assert Project("hub://<tenant>/<name>").name == "<name>"

    def test_put(self): ...

    def test_put_exception(self): ...

    def test_reports(self): ...

    def test_reports_get(self): ...

    def test_reports_metadata(self): ...

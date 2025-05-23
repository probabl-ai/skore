from sys import version_info
from unittest.mock import Mock

from pandas import DataFrame, MultiIndex, Series
from pytest import fixture, raises
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport, Project
from skore.project.metadata import Metadata

if version_info < (3, 10):
    from importlib_metadata import EntryPoint, EntryPoints
else:
    from importlib.metadata import EntryPoint, EntryPoints


class FakeLocalProject(Mock):
    def __init__(self, *args, **kwargs):
        super().__init__(constructor_args=args, constructor_kwargs=kwargs)


class FakeHubProject(Mock):
    def __init__(self, *args, **kwargs):
        super().__init__(constructor_args=args, constructor_kwargs=kwargs)


class TestProject:
    @fixture(autouse=True)
    def monkeypatch_entrypoints(self, monkeypatch, request):
        monkeypatch.setattr(
            "skore.project.project.entry_points",
            lambda **kwargs: EntryPoints(
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
            ),
        )

    @fixture(scope="class")
    def regression(self):
        X, y = make_regression(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return EstimatorReport(
            LinearRegression(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

    def test_init_local(self):
        project = Project("<name>")

        assert isinstance(project, Project)
        assert project._Project__mode == "local"
        assert project._Project__name == "<name>"
        assert isinstance(project._Project__project, FakeLocalProject)
        assert not project._Project__project.constructor_args
        assert project._Project__project.constructor_kwargs == {"name": "<name>"}

    def test_init_hub(self):
        project = Project("hub://<tenant>/<name>")

        assert isinstance(project, Project)
        assert project._Project__mode == "hub"
        assert project._Project__name == "<name>"
        assert isinstance(project._Project__project, FakeHubProject)
        assert not project._Project__project.constructor_args
        assert project._Project__project.constructor_kwargs == {
            "tenant": "<tenant>",
            "name": "<name>",
        }

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

    def test_put(self, regression):
        project = Project("<name>")

        project.put("<key>", regression)

        assert isinstance(project._Project__project, FakeLocalProject)
        assert project._Project__project.put.called
        assert not project._Project__project.put.call_args.args
        assert project._Project__project.put.call_args.kwargs == {
            "key": "<key>",
            "report": regression,
        }

    def test_put_exception(self):
        with raises(TypeError, match="Key must be a string"):
            Project("<name>").put(None, "<value>")

        with raises(TypeError, match="Report must be a `skore.EstimatorReport`"):
            Project("<name>").put("<key>", "<value>")

    def test_reports(self):
        project = Project("<name>")

        assert hasattr(project, "reports")
        assert hasattr(project.reports, "get")
        assert hasattr(project.reports, "metadata")

    def test_reports_get(self):
        project = Project("<name>")

        project.reports.get("<id>")

        assert isinstance(project._Project__project, FakeLocalProject)
        assert project._Project__project.reports.get.called
        assert project._Project__project.reports.get.call_args.args == ("<id>",)
        assert not project._Project__project.reports.get.call_args.kwargs

    def test_reports_metadata(self):
        project = Project("<name>")
        project._Project__project.reports.metadata.return_value = [
            {
                "learner": "<learner>",
                "accuracy": 1.0,
                "id": "<id>",
            }
        ]

        metadata = project.reports.metadata()

        assert project._Project__project.reports.metadata.called
        assert isinstance(metadata, DataFrame)
        assert isinstance(metadata, Metadata)
        assert DataFrame.equals(
            metadata,
            DataFrame(
                data={
                    "learner": Series(
                        ["<learner>"],
                        dtype="category",
                        index=[(0, "<id>")],
                    ),
                    "accuracy": Series([1.0], index=[(0, "<id>")]),
                },
                index=MultiIndex.from_tuples([(0, "<id>")], names=[None, "id"]),
            ),
        )

    def test_repr(self):
        project = Project("<name>")
        assert repr(project) == repr(project._Project__project)

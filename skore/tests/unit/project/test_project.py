from sys import version_info
from unittest.mock import Mock

from pandas import DataFrame, MultiIndex, Series
from pytest import fixture, raises
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skore import EstimatorReport, Project
from skore.project.summary import Summary

if version_info < (3, 10):
    from importlib_metadata import EntryPoint, EntryPoints
else:
    from importlib.metadata import EntryPoint, EntryPoints


class FakeEntryPoint(EntryPoint):
    def load(self):
        return self.value


@fixture
def FakeLocalProject():
    return Mock()


@fixture
def FakeHubProject():
    return Mock()


@fixture(autouse=True)
def monkeypatch_entrypoints(monkeypatch, request, FakeLocalProject, FakeHubProject):
    monkeypatch.setattr(
        "skore.project.project.entry_points",
        lambda **kwargs: EntryPoints(
            [
                FakeEntryPoint(
                    name="local",
                    value=FakeLocalProject,
                    group="skore.plugins.project",
                ),
                FakeEntryPoint(
                    name="hub",
                    value=FakeHubProject,
                    group="skore.plugins.project",
                ),
            ]
        ),
    )


@fixture(scope="module")
def regression():
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


class TestProject:
    def test_init_local(self, FakeLocalProject):
        project = Project("<name>", workspace="<workspace>")

        assert isinstance(project, Project)
        assert project._Project__mode == "local"
        assert project._Project__name == "<name>"
        assert FakeLocalProject.called
        assert not FakeLocalProject.call_args.args
        assert FakeLocalProject.call_args.kwargs == {
            "name": "<name>",
            "workspace": "<workspace>",
        }

    def test_init_hub(self, FakeHubProject):
        project = Project("hub://<tenant>/<name>")

        assert isinstance(project, Project)
        assert project._Project__mode == "hub"
        assert project._Project__name == "<name>"
        assert FakeHubProject.called
        assert not FakeHubProject.call_args.args
        assert FakeHubProject.call_args.kwargs == {
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

    def test_put(self, regression, FakeLocalProject):
        project = Project("<name>")

        project.put("<key>", regression)

        assert FakeLocalProject.called
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

    def test_get(self, FakeLocalProject):
        project = Project("<name>")

        project.get("<id>")

        assert FakeLocalProject.called
        assert project._Project__project.reports.get.called
        assert project._Project__project.reports.get.call_args.args == ("<id>",)
        assert not project._Project__project.reports.get.call_args.kwargs

    def test_summary(self):
        project = Project("<name>")
        project._Project__project.reports.metadata.return_value = [
            {
                "learner": "<learner>",
                "accuracy": 1.0,
                "id": "<id>",
            }
        ]

        summary = project.summary()

        assert project._Project__project.reports.metadata.called
        assert isinstance(summary, DataFrame)
        assert isinstance(summary, Summary)
        assert DataFrame.equals(
            summary,
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

    def test_delete_local(self, FakeLocalProject):
        Project.delete("<name>", workspace="<workspace>")

        assert not FakeLocalProject.called
        assert FakeLocalProject.delete.called
        assert not FakeLocalProject.delete.call_args.args
        assert FakeLocalProject.delete.call_args.kwargs == {
            "name": "<name>",
            "workspace": "<workspace>",
        }

    def test_delete_hub(self, FakeHubProject):
        Project.delete("hub://<tenant>/<name>")

        assert not FakeHubProject.called
        assert FakeHubProject.delete.called
        assert not FakeHubProject.delete.call_args.args
        assert FakeHubProject.delete.call_args.kwargs == {
            "tenant": "<tenant>",
            "name": "<name>",
        }

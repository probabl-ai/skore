from importlib.metadata import EntryPoint, EntryPoints
from re import escape
from unittest.mock import Mock

from pandas import DataFrame, MultiIndex, Series
from pytest import fixture, mark, param, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from skore import CrossValidationReport, EstimatorReport, Project
from skore.project.summary import Summary


class FakeEntryPoint(EntryPoint):
    def load(self):
        return self.value


@fixture
def FakeLocalProject():
    project = Mock()
    project.summarize = Mock(return_value=[])
    project_factory = Mock(return_value=project)
    return project_factory


@fixture
def FakeHubProject():
    project = Mock()
    project.summarize = Mock(return_value=[])
    project_factory = Mock(return_value=project)
    return project_factory


@fixture(autouse=True)
def monkeypatch_entrypoints(monkeypatch, FakeLocalProject, FakeHubProject):
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
def regression() -> EstimatorReport:
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


@fixture(scope="module")
def classification() -> EstimatorReport:
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return EstimatorReport(
        LogisticRegression(),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


@fixture(scope="module")
def cv_regression() -> CrossValidationReport:
    return CrossValidationReport(LinearRegression(), *make_regression(random_state=42))


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
        project = Project("hub://<workspace>/<name>")

        assert isinstance(project, Project)
        assert project._Project__mode == "hub"
        assert project._Project__name == "<name>"
        assert FakeHubProject.called
        assert not FakeHubProject.call_args.args
        assert FakeHubProject.call_args.kwargs == {
            "workspace": "<workspace>",
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
            match=escape("Unknown mode `local`. Please install `skore[local]`."),
        ):
            Project("<name>")

    def test_init_exception_wrong_ml_task(self, monkeypatch):
        """If the underlying Project implementation contains reports with
        different ML tasks, the top-level `skore.Project` will raise."""

        project = Mock()
        project.summarize = Mock(
            return_value=[
                {"ml_task": "binary-classification"},
                {"ml_task": "regression"},
            ]
        )
        project_factory = Mock(return_value=project)

        monkeypatch.setattr(
            "skore.project.project.entry_points",
            lambda **kwargs: EntryPoints(
                [
                    FakeEntryPoint(
                        name="local",
                        value=project_factory,
                        group="skore.plugins.project",
                    ),
                ]
            ),
        )

        err_msg = (
            "Expected every report in the Project to have the same ML task. "
            "Got ML tasks "
        )
        with raises(RuntimeError, match=err_msg):
            Project("<name>", workspace="<workspace>")

    def test_mode(self):
        assert Project("<name>").mode == "local"
        assert Project("hub://<workspace>/<name>").mode == "hub"

    def test_name(self):
        assert Project("<name>").name == "<name>"
        assert Project("hub://<workspace>/<name>").name == "<name>"

    @mark.parametrize(
        "report",
        (
            param("regression", id="EstimatorReport - regression"),
            param("cv_regression", id="CrossValidationReport - regression"),
        ),
    )
    def test_put(self, report, FakeLocalProject, request):
        report = request.getfixturevalue(report)
        project = Project("<name>")

        project.put("<key>", report)

        assert FakeLocalProject.called
        assert project._Project__project.put.called
        assert not project._Project__project.put.call_args.args
        assert project._Project__project.put.call_args.kwargs == {
            "key": "<key>",
            "report": report,
        }

    def test_put_exception(self):
        with raises(TypeError, match="Key must be a string"):
            Project("<name>").put(None, "<value>")

        with raises(TypeError, match="Report must be `EstimatorReport` or"):
            Project("<name>").put("<key>", "<value>")

    def test_put_exception_wrong_ml_task(self, regression, classification):
        project = Project("<name>", workspace="<workspace>")
        project.put("classification", classification)
        assert project.ml_task == "binary-classification"

        err_msg = (
            "Expected a report meant for ML task 'binary-classification' "
            "but the given report is for ML task 'regression'"
        )
        with raises(ValueError, match=err_msg):
            project.put("regression", regression)

    def test_get(self, FakeLocalProject):
        project = Project("<name>")

        project.get("<id>")

        assert FakeLocalProject.called
        assert project._Project__project.get.called
        assert project._Project__project.get.call_args.args == ("<id>",)
        assert not project._Project__project.get.call_args.kwargs

    def test_summarize(self):
        project = Project("<name>")
        project._Project__project.summarize.return_value = [
            {
                "learner": "<learner>",
                "accuracy": 1.0,
                "id": "<id>",
            }
        ]

        summary = project.summarize()

        assert project._Project__project.summarize.called
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

    def test_summarize_with_skore_local_project(self, monkeypatch, tmpdir):
        """Smoke test to check that ModelExplorerWidget can be shown."""
        from IPython.core.interactiveshell import InteractiveShell

        snippet = f"""
        from pathlib import Path

        from sklearn.datasets import make_regression
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from skore import EstimatorReport, Project

        X, y = make_regression(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        regression = EstimatorReport(
            LinearRegression(),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        project = Project("<project>", workspace=Path(r"{tmpdir}"))
        project.put("<report>", regression)
        project.summarize()
        """

        monkeypatch.undo()

        shell = InteractiveShell.instance()
        execution_result = shell.run_cell(snippet, silent=True)
        execution_result.raise_error()

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
        Project.delete("hub://<workspace>/<name>")

        assert not FakeHubProject.called
        assert FakeHubProject.delete.called
        assert not FakeHubProject.delete.call_args.args
        assert FakeHubProject.delete.call_args.kwargs == {
            "workspace": "<workspace>",
            "name": "<name>",
        }

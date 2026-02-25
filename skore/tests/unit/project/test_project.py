from importlib.metadata import EntryPoint, EntryPoints
from re import escape
from unittest.mock import Mock

from pandas import DataFrame, MultiIndex, Series
from pytest import fixture, mark, param, raises
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from skore import CrossValidationReport, EstimatorReport, Project
from skore._project._summary import Summary


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


@fixture
def FakeMlflowProject():
    project = Mock()
    project.summarize = Mock(return_value=[])
    project_factory = Mock(return_value=project)
    return project_factory


@fixture(autouse=True)
def monkeypatch_entrypoints(
    monkeypatch, FakeLocalProject, FakeHubProject, FakeMlflowProject
):
    monkeypatch.setattr(
        "skore._project.plugin.entry_points",
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
                FakeEntryPoint(
                    name="mlflow",
                    value=FakeMlflowProject,
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
        project = Project(mode="local", name="<name>", workspace="<workspace>")

        assert isinstance(project, Project)
        assert project._Project__mode == "local"
        assert project._Project__name == "<name>"
        assert FakeLocalProject.called
        assert not FakeLocalProject.call_args.args
        assert FakeLocalProject.call_args.kwargs == {
            "name": "<name>",
            "workspace": "<workspace>",
        }

    def test_init_local_unknown_plugin(self, monkeypatch, tmp_path):
        monkeypatch.undo()
        monkeypatch.setattr(
            "skore._project.plugin.entry_points", lambda **kwargs: EntryPoints([])
        )

        with raises(
            ValueError,
            match=escape(
                "The mode `local` is not supported. You need to install "
                "`skore-local-project` to use it."
            ),
        ):
            Project(mode="local", name="<name>")

    def test_init_hub(self, FakeHubProject):
        project = Project(mode="hub", name="<workspace>/<name>")

        assert isinstance(project, Project)
        assert project._Project__mode == "hub"
        assert project._Project__name == "<workspace>/<name>"
        assert FakeHubProject.called
        assert not FakeHubProject.call_args.args
        assert FakeHubProject.call_args.kwargs == {
            "workspace": "<workspace>",
            "name": "<name>",
        }

    def test_init_mlflow(self, FakeMlflowProject):
        project = Project(mode="mlflow", name="<name>", tracking_uri="<uri>")

        assert isinstance(project, Project)
        assert project._Project__mode == "mlflow"
        assert project._Project__name == "<name>"
        assert FakeMlflowProject.called
        assert not FakeMlflowProject.call_args.args
        assert FakeMlflowProject.call_args.kwargs == {
            "name": "<name>",
            "tracking_uri": "<uri>",
        }

    def test_init_hub_unknown_plugin(self, monkeypatch, tmp_path):
        monkeypatch.undo()
        monkeypatch.setattr(
            "skore._project.plugin.entry_points", lambda **kwargs: EntryPoints([])
        )

        with raises(
            ValueError,
            match=escape(
                "The mode `hub` is not supported. You need to install "
                "`skore-hub-project` to use it."
            ),
        ):
            Project(mode="hub", name="<workspace>/<name>")

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
            "skore._project.plugin.entry_points",
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
            Project(mode="local", name="<name>", workspace="<workspace>")

    def test_mode(self):
        assert Project(mode="local", name="<name>").mode == "local"
        assert Project(mode="hub", name="<workspace>/<name>").mode == "hub"
        assert Project(mode="mlflow", name="<name>").mode == "mlflow"

    def test_name(self):
        assert Project(mode="local", name="<name>").name == "<name>"
        assert (
            Project(mode="hub", name="<workspace>/<name>").name == "<workspace>/<name>"
        )

    @mark.parametrize(
        "report",
        (
            param("regression", id="EstimatorReport - regression"),
            param("cv_regression", id="CrossValidationReport - regression"),
        ),
    )
    def test_put(self, report, FakeLocalProject, request):
        report = request.getfixturevalue(report)
        project = Project(mode="local", name="<name>")

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
            Project(mode="local", name="<name>").put(None, "<value>")

        with raises(TypeError, match="Report must be `EstimatorReport` or"):
            Project(mode="local", name="<name>").put("<key>", "<value>")

    def test_put_exception_wrong_ml_task(self, regression, classification):
        project = Project(mode="local", name="<name>", workspace="<workspace>")
        project.put("classification", classification)
        assert project.ml_task == "binary-classification"

        err_msg = (
            "At this time, a Project can only contain reports associated with a single "
            "ML task. Project '<name>' expected ML task 'binary-classification'; got a "
            "report associated with ML task 'regression'."
        )
        with raises(ValueError, match=err_msg):
            project.put("regression", regression)

    def test_get(self, FakeLocalProject):
        project = Project(mode="local", name="<name>")

        project.get("<id>")

        assert FakeLocalProject.called
        assert project._Project__project.get.called
        assert project._Project__project.get.call_args.args == ("<id>",)
        assert not project._Project__project.get.call_args.kwargs

    def test_summarize(self):
        project = Project(mode="local", name="<name>")
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

        project = Project(mode="local", name="<project>", workspace=Path(r"{tmpdir}"))
        project.put("<report>", regression)
        project.summarize()
        """

        monkeypatch.undo()

        shell = InteractiveShell.instance()
        execution_result = shell.run_cell(snippet, silent=True)
        execution_result.raise_error()

    def test_repr(self):
        project = Project(mode="local", name="<name>")
        assert repr(project) == repr(project._Project__project)

    def test_delete_local(self, FakeLocalProject):
        Project.delete(mode="local", name="<name>", workspace="<workspace>")

        assert not FakeLocalProject.called
        assert FakeLocalProject.delete.called
        assert not FakeLocalProject.delete.call_args.args
        assert FakeLocalProject.delete.call_args.kwargs == {
            "name": "<name>",
            "workspace": "<workspace>",
        }

    def test_delete_hub(self, FakeHubProject):
        Project.delete(mode="hub", name="<workspace>/<name>")

        assert not FakeHubProject.called
        assert FakeHubProject.delete.called
        assert not FakeHubProject.delete.call_args.args
        assert FakeHubProject.delete.call_args.kwargs == {
            "workspace": "<workspace>",
            "name": "<name>",
        }

    def test_delete_mlflow(self, FakeMlflowProject):
        Project.delete(mode="mlflow", name="<name>", tracking_uri="<uri>")

        assert not FakeMlflowProject.called
        assert FakeMlflowProject.delete.called
        assert not FakeMlflowProject.delete.call_args.args
        assert FakeMlflowProject.delete.call_args.kwargs == {
            "name": "<name>",
            "tracking_uri": "<uri>",
        }

    def test_delete_mlflow_with_unexpected_kwargs(self, FakeMlflowProject):
        with raises(TypeError, match="Unexpected keyword argument\\(s\\): 'workspace'"):
            Project.delete(
                mode="mlflow",
                name="<name>",
                tracking_uri="<uri>",
                workspace="<workspace>",
            )

        assert not FakeMlflowProject.called
        assert not FakeMlflowProject.delete.called

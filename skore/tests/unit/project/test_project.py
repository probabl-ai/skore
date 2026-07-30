from importlib.metadata import EntryPoint, EntryPoints
from re import escape
from unittest.mock import Mock
from uuid import uuid4

from pandas import Timestamp
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
    def factory(**kwargs):
        project = Mock()
        project.name = kwargs.get("name", "<name>")
        if "workspace" in kwargs:
            project.workspace = kwargs["workspace"]
        project.summarize = Mock(return_value=[])
        return project

    return Mock(side_effect=factory)


@fixture
def FakeHubProject():
    def factory(**kwargs):
        project = Mock()
        project.name = kwargs.get("name", "<name>")
        project.workspace = kwargs.get("workspace", "<workspace>")
        project.summarize = Mock(return_value=[])
        return project

    return Mock(side_effect=factory)


@fixture
def FakeMlflowProject():
    def factory(**kwargs):
        project = Mock()
        project.name = kwargs.get("name", "<name>")
        project.tracking_uri = kwargs.get("tracking_uri", "file:///tmp/mlflow")
        project.summarize = Mock(return_value=[])
        return project

    return Mock(side_effect=factory)


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
        project = Project(name="<name>", mode="local", workspace="<workspace>")

        assert isinstance(project, Project)
        assert project.mode == "local"
        assert project.name == "<name>"
        assert project.workspace == "<workspace>"
        assert project.tracking_uri is None
        assert FakeLocalProject.called
        assert not FakeLocalProject.call_args.args
        assert FakeLocalProject.call_args.kwargs == {
            "name": "<name>",
            "workspace": "<workspace>",
        }

    def test_init_local_missing_optional_dependency(self, monkeypatch):
        fake_library_name = uuid4().hex

        def fake_requires(name):
            assert name == "skore"
            return [f'{fake_library_name} ; extra == "local"']

        monkeypatch.setattr("skore._project.dependencies.requires", fake_requires)

        with raises(
            ImportError,
            match=escape(
                f"Missing library: `{fake_library_name}`. "
                "You can fix this error by installing `skore[local]`."
            ),
        ):
            Project(name="<name>", mode="local")

    def test_init_local_missing_optional_dependency_without_plugin(self, monkeypatch):
        """Non-regression test for missing extras before plugin discovery."""
        fake_library_name = uuid4().hex

        def fake_requires(name):
            assert name == "skore"
            return [f'{fake_library_name} ; extra == "local"']

        monkeypatch.undo()
        monkeypatch.setattr("skore._project.dependencies.requires", fake_requires)
        monkeypatch.setattr(
            "skore._project.plugin.entry_points", lambda **kwargs: EntryPoints([])
        )

        with raises(
            ImportError,
            match=escape(
                f"Missing library: `{fake_library_name}`. "
                "You can fix this error by installing `skore[local]`."
            ),
        ):
            Project(name="<name>", mode="local")

    def test_init_hub(self, FakeHubProject, monkeypatch):
        monkeypatch.setattr("skore._project.dependencies.requires", lambda _: [])

        project = Project(name="<name>", mode="hub", workspace="<workspace>")

        assert isinstance(project, Project)
        assert project.mode == "hub"
        assert project.name == "<name>"
        assert project.workspace == "<workspace>"
        assert project.tracking_uri is None
        assert FakeHubProject.called
        assert not FakeHubProject.call_args.args
        assert FakeHubProject.call_args.kwargs == {
            "name": "<name>",
            "workspace": "<workspace>",
        }

    def test_init_mlflow(self, FakeMlflowProject, monkeypatch):
        monkeypatch.setattr("skore._project.dependencies.requires", lambda _: [])

        project = Project(name="<name>", mode="mlflow", tracking_uri="<uri>")

        assert isinstance(project, Project)
        assert project.mode == "mlflow"
        assert project.name == "<name>"
        assert project.workspace is None
        assert project.tracking_uri == "<uri>"
        assert FakeMlflowProject.called
        assert not FakeMlflowProject.call_args.args
        assert FakeMlflowProject.call_args.kwargs == {
            "name": "<name>",
            "tracking_uri": "<uri>",
        }

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
            Project(name="<name>", mode="local", workspace="<workspace>")

    def test_mode(self, monkeypatch):
        monkeypatch.setattr("skore._project.dependencies.requires", lambda _: [])

        assert Project(name="name", mode="local").mode == "local"
        assert Project(name="name", mode="hub", workspace="workspace").mode == "hub"
        assert Project(name="name", mode="mlflow").mode == "mlflow"

    def test_name(self, monkeypatch):
        monkeypatch.setattr("skore._project.dependencies.requires", lambda _: [])

        assert Project(name="name", mode="local").name == "name"
        assert Project(name="name", mode="hub", workspace="workspace").name == "name"
        assert Project(name="name", mode="mlflow").name == "name"

    @mark.parametrize(
        "report",
        (
            param("regression", id="EstimatorReport - regression"),
            param("cv_regression", id="CrossValidationReport - regression"),
        ),
    )
    def test_put(self, report, FakeLocalProject, request):
        report = request.getfixturevalue(report)
        project = Project(name="<name>", mode="local")

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
            Project(name="<name>", mode="local").put(None, "<value>")

        with raises(TypeError, match="Report must be `EstimatorReport` or"):
            Project(name="<name>", mode="local").put("<key>", "<value>")

    def test_put_exception_wrong_ml_task(self, regression, classification):
        project = Project(name="<name>", mode="local", workspace="<workspace>")
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
        project = Project(name="<name>", mode="local")

        project.get("<id>")

        assert FakeLocalProject.called
        assert project._Project__project.get.called
        assert project._Project__project.get.call_args.args == ("<id>",)
        assert not project._Project__project.get.call_args.kwargs

    def test_summarize_sorts_by_date(self):
        project = Project(name="<name>", mode="local")
        project._Project__project.summarize.return_value = [
            {
                "id": "<id-2>",
                "key": "<key-2>",
                "date": "2024-01-02T00:00:00",
                "learner": "<learner>",
                "ml_task": "regression",
                "report_type": "estimator",
                "dataset": "<dataset>",
            },
            {
                "id": "<id-1>",
                "key": "<key-1>",
                "date": "2024-01-01T00:00:00",
                "learner": "<learner>",
                "ml_task": "regression",
                "report_type": "estimator",
                "dataset": "<dataset>",
            },
        ]

        summary = project.summarize()

        assert summary.frame().index.get_level_values("id").tolist() == [
            "<id-1>",
            "<id-2>",
        ]

    def test_summarize(self):
        project = Project(name="<name>", mode="local")
        project._Project__project.summarize.return_value = [
            {
                "id": "<id>",
                "key": "<key>",
                "date": "2024-01-01T00:00:00",
                "learner": "<learner>",
                "ml_task": "regression",
                "report_type": "estimator",
                "dataset": "<dataset>",
                "accuracy": 1.0,
            }
        ]

        summary = project.summarize()

        assert project._Project__project.summarize.called
        assert isinstance(summary, Summary)
        assert summary.frame().to_dict() == {
            "accuracy": {
                (0, "<id>"): 1.0,
            },
            "dataset": {
                (0, "<id>"): "<dataset>",
            },
            "date": {
                (0, "<id>"): Timestamp("2024-01-01 00:00:00"),
            },
            "key": {
                (0, "<id>"): "<key>",
            },
            "learner": {
                (0, "<id>"): "<learner>",
            },
            "report_type": {
                (0, "<id>"): "estimator",
            },
        }

    def test_summarize_with_skore_local_project(self, monkeypatch, tmpdir):
        """Smoke test to check that the summary HTML repr can be shown."""
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

        project = Project(name="<project>", mode="local", workspace=Path(r"{tmpdir}"))
        project.put("<report>", regression)
        summary = project.summarize()
        summary._repr_mimebundle_()
        """

        monkeypatch.undo()

        shell = InteractiveShell.instance()
        execution_result = shell.run_cell(snippet, silent=True)
        execution_result.raise_error()

    def test_repr(self):
        project = Project(name="<name>", mode="local")
        assert repr(project) == repr(project._Project__project)

    def test_delete_local(self, FakeLocalProject):
        Project.delete(name="<name>", mode="local", workspace="<workspace>")

        assert not FakeLocalProject.called
        assert FakeLocalProject.delete.called
        assert not FakeLocalProject.delete.call_args.args
        assert FakeLocalProject.delete.call_args.kwargs == {
            "name": "<name>",
            "workspace": "<workspace>",
        }

    def test_delete_hub(self, FakeHubProject):
        Project.delete(name="<name>", mode="hub", workspace="<workspace>")

        assert not FakeHubProject.called
        assert FakeHubProject.delete.called
        assert not FakeHubProject.delete.call_args.args
        assert FakeHubProject.delete.call_args.kwargs == {
            "name": "<name>",
            "workspace": "<workspace>",
        }

    def test_delete_mlflow(self, FakeMlflowProject):
        Project.delete(name="<name>", mode="mlflow", tracking_uri="<uri>")

        assert not FakeMlflowProject.called
        assert FakeMlflowProject.delete.called
        assert not FakeMlflowProject.delete.call_args.args
        assert FakeMlflowProject.delete.call_args.kwargs == {
            "name": "<name>",
            "tracking_uri": "<uri>",
        }

    def test_delete_mlflow_forwards_kwargs(self, FakeMlflowProject):
        Project.delete(
            name="<name>",
            mode="mlflow",
            tracking_uri="<uri>",
            workspace="<workspace>",
        )

        assert not FakeMlflowProject.called
        assert FakeMlflowProject.delete.called
        assert not FakeMlflowProject.delete.call_args.args
        assert FakeMlflowProject.delete.call_args.kwargs == {
            "name": "<name>",
            "tracking_uri": "<uri>",
            "workspace": "<workspace>",
        }

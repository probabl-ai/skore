"""Class definition of the ``skore`` project."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from pandas import DataFrame, Index, MultiIndex, RangeIndex

from skore._project import plugin
from skore._project._summary import Summary
from skore._project.dependencies import assert_optional_dependencies_installed
from skore._project.types import ProjectMode

if TYPE_CHECKING:
    from skore import CrossValidationReport, EstimatorReport


MODES = get_args(ProjectMode)


class Project:
    r"""
    API to manage a collection of key-report pairs.

    Its constructor initializes a project by creating a new project or by loading an
    existing one.

    The class main methods are :func:`~skore.Project.put`,
    :func:`~skore.Project.summarize` and :func:`~skore.Project.get`, respectively to
    insert a key-report pair into the project, to obtain the metadata/metrics of the
    inserted reports and to get a specific report by its id.

    Three mutually exclusive modes are available and can be configured using the
    ``mode`` parameter of the constructor:

    .. rubric:: Hub mode

    The project is configured to communicate with ``skore hub``.

    In this mode, ``workspace`` is a ``skore hub`` concept that must be configured on
    the ``skore hub`` interface. It represents an isolated entity managing users,
    projects, and resources. It can be a company, organization, or team that operates
    independently within the system.

    Note: Using Project in ``hub`` mode requires an account on ``skore hub``, with
    access rights to the specified workspace. Authentication to ``skore hub`` is done by
    running ``skore.login()`` before instantiating the Project.

    .. rubric:: Local mode

    Otherwise, the project is configured to the ``local`` mode to be persisted on
    the user machine in a directory called ``workspace``.

    | The workspace can be shared between all the projects.
    | The workspace can be set using kwargs or the environment variable
      ``SKORE_WORKSPACE``.
    | If not, it will be by default set to a ``skore/`` directory in the user
      data directory:

    - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
    - on Linux, usually ``${HOME}/.local/share/skore``,
    - on macOS, usually ``${HOME}/Library/Application Support/skore``.

    .. rubric:: MLflow mode

    In this mode, ``name`` is used as the MLflow experiment name. Reports are persisted
    as MLflow model artifacts in runs created under this experiment.

    Refer to the :ref:`project` section of the user guide for more details.

    Parameters
    ----------
    name : str
        The name of the project.
    mode : {"hub", "local", "mlflow"}
        The mode of the project.
    **kwargs : dict
        Extra keyword arguments passed to the project, depending on its mode.

        workspace : str or Path-like, optional

            - If ``mode="hub"``, the Hub workspace name (required);
            - If ``mode="local"``, the local persistence directory (optional);
            - If ``mode="mlflow"``, ignored.

        tracking_uri : str, mode:mlflow only.
            The URI of the MLflow tracking server.

    Attributes
    ----------
    name : str
        The name of the project.
    mode : {"hub", "local", "mlflow"}
        The mode of the project.
    workspace : Path or str or None
        The workspace for ``local`` (``Path``) and ``hub`` (``str``) modes; ``None``
        otherwise.
    tracking_uri : str or None
        The MLflow tracking URI for ``mlflow`` mode; ``None`` otherwise.
    ml_task : MLTask
        The ML task of the project; unset until a first report is put.

    Examples
    --------
    Construct reports.

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import LinearRegression
    >>> from skore import evaluate
    >>>
    >>> X, y = make_regression(random_state=42)
    >>> regressor = LinearRegression()
    >>> regressor_report = evaluate(regressor, X, y, splitter=0.2)

    Construct the project in local mode, persisted in a temporary directory.

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from skore import Project
    >>>
    >>> tmpdir = TemporaryDirectory().name
    >>> local_project = Project(name="my-xp", mode="local", workspace=Path(tmpdir))

    Put reports in the project.

    >>> local_project.put("my-simple-regression", regressor_report)

    Investigate metadata/metrics to filter the best reports.

    >>> summary = local_project.summarize()
    >>> summary = summary.query("rmse < 67")
    >>> reports = summary.compare()

    See Also
    --------
    :class:`~skore.Summary` :
        Tabular view of metadata and metrics for persisted reports.
    :func:`~skore.compare` :
        Compare reports side by side.
    :meth:`Project.summarize` :
        Create a summary view to investigate persisted reports' metadata/metrics.
    """

    @staticmethod
    def __setup_plugin(
        mode: ProjectMode, name: str, **kwargs: Any
    ) -> tuple[Any, dict[str, Any]]:
        if mode not in MODES:
            raise ValueError(f"`mode` must be included in {MODES} (found {mode})")

        assert_optional_dependencies_installed(mode)

        return (
            plugin.get(group="skore.plugins.project", mode=mode),
            {"name": name} | kwargs,
        )

    def __init__(self, name: str, *, mode: ProjectMode = "local", **kwargs):
        r"""
        Initialize a project.

        Parameters
        ----------
        name : str
            The name of the project.
            For mode:mlflow, this name will be used as the experiment name.
        mode : {"hub", "local", "mlflow"}, default "local"
            The mode of the project.
        **kwargs : dict
            Extra keyword arguments passed to the project, depending on its mode.

            workspace : str or Path-like, optional
                Hub workspace name when ``mode="hub"`` (required). Local persistence
                directory when ``mode="local"`` (optional). Ignored when
                ``mode="mlflow"``.

                For ``mode="local"``:

                | The workspace can be shared between all the projects.
                | The workspace can be set using kwargs or the environment variable
                  ``SKORE_WORKSPACE``.
                | If not, it will be by default set to a ``skore/`` directory in the
                  user data directory:

                - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
                - on Linux, usually ``${HOME}/.local/share/skore``,
                - on macOS, usually ``${HOME}/Library/Application Support/skore``.

            tracking_uri : str, mode:mlflow only.
                The URI of the MLflow tracking server.

        Examples
        --------
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> from skore import Project
        >>> tmpdir = TemporaryDirectory()
        >>> project = Project(name="my-xp", mode="local", workspace=Path(tmpdir.name))
        >>> project.name
        'my-xp'
        >>> project.mode
        'local'
        >>> tmpdir.cleanup()
        """
        plugin, parameters = Project.__setup_plugin(mode, name, **kwargs)

        self.__mode = mode
        self.__project = plugin(**parameters)

        ml_tasks = {report["ml_task"] for report in self.__project.summarize()}

        if len(ml_tasks) > 1:
            raise RuntimeError(
                "Expected every report in the Project to have the same ML task. "
                f"Got ML tasks {ml_tasks}."
            )

        self.ml_task = ml_tasks.pop() if ml_tasks else None

    @property
    def mode(self) -> ProjectMode:
        """The mode of the project."""
        return self.__mode

    @property
    def name(self) -> str:
        """The name of the project."""
        return self.__project.name

    @property
    def workspace(self) -> Path | str | None:
        """The workspace for local and hub modes; ``None`` otherwise."""
        if self.__mode in ("local", "hub"):
            return self.__project.workspace
        return None

    @property
    def tracking_uri(self) -> str | None:
        """The MLflow tracking URI for mlflow mode; ``None`` otherwise."""
        if self.__mode == "mlflow":
            return self.__project.tracking_uri
        return None

    def put(self, key: str, report: EstimatorReport | CrossValidationReport):
        """
        Put a key-report pair to the project.

        If the key already exists, its last report is modified to point to this new
        report, while keeping track of the report history.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the project.
            Name of the run for mode:mlflow
        report : EstimatorReport | CrossValidationReport
            The report to associate with ``key`` in the project.

        Returns
        -------
        None
            The report is persisted in the project backend.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import LinearRegression
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> from skore import Project, evaluate
        >>> X, y = make_regression(random_state=42)
        >>> report = evaluate(LinearRegression(), X, y, splitter=0.2)
        >>> tmpdir = TemporaryDirectory()
        >>> project = Project(name="my-xp", mode="local", workspace=Path(tmpdir.name))
        >>> project.put("my-regression", report)
        >>> tmpdir.cleanup()
        """
        from skore import CrossValidationReport, EstimatorReport

        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        if not isinstance(report, EstimatorReport | CrossValidationReport):
            raise TypeError(
                f"Report must be `EstimatorReport` or `CrossValidationReport` "
                f"(found '{type(report)}')"
            )

        if self.ml_task is not None:
            if report.ml_task != self.ml_task:
                raise ValueError(
                    "At this time, a Project can only contain reports associated with "
                    "a single ML task. "
                    f"Project {self.name!r} expected ML task {self.ml_task!r}; "
                    f"got a report associated with ML task {report.ml_task!r}."
                )
        else:
            self.ml_task = report.ml_task

        return self.__project.put(key=key, report=report)

    def get(self, id: str) -> EstimatorReport | CrossValidationReport:
        """
        Get a persisted report by its id.

        Report IDs can be found via :meth:`skore.Project.summarize`, which is also the
        preferred method of interacting with a ``skore.Project``. The ``id`` passed here
        must match the ``id`` column returned by :meth:`Project.summarize`.

        Parameters
        ----------
        id : str
            The id of a report already put in the ``project``.

        Returns
        -------
        report : EstimatorReport or CrossValidationReport
            The report associated with ``id``.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import LinearRegression
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> from skore import Project, evaluate
        >>> X, y = make_regression(random_state=42)
        >>> report = evaluate(LinearRegression(), X, y, splitter=0.2)
        >>> tmpdir = TemporaryDirectory()
        >>> project = Project(name="my-xp", mode="local", workspace=Path(tmpdir.name))
        >>> project.put("my-regression", report)
        >>> summary = project.summarize()
        >>> report_id = summary.frame().index.get_level_values("id")[0]
        >>> retrieved = project.get(report_id)
        >>> type(retrieved).__name__
        'EstimatorReport'
        >>> tmpdir.cleanup()
        """
        return self.__project.get(id)

    def summarize(self) -> Summary:
        """Obtain metadata/metrics for all persisted reports.

        Reports are returned in ascending order of their ``date`` field.

        Returns
        -------
        summary : Summary
            Metadata and metrics for every report persisted in the project.

        See Also
        --------
        :class:`~skore.Summary` :
            Tabular view with interactive filtering in Jupyter.
        :func:`~skore.compare` :
            Compare selected reports side by side.
        """
        records = self.__project.summarize()
        if records:
            records = sorted(records, key=lambda record: record["date"])
        frame = DataFrame(records, copy=False)
        if not frame.empty:
            frame.index = MultiIndex.from_arrays(
                [
                    RangeIndex(len(frame)),
                    Index(frame.pop("id"), name="id", dtype=str),
                ]
            )
        return Summary(frame, self.__project)

    def __repr__(self) -> str:  # noqa: D105
        return self.__project.__repr__()

    @staticmethod
    def delete(name: str, *, mode: ProjectMode = "local", **kwargs):
        r"""
        Delete a project.

        Parameters
        ----------
        name : str
            The name of the project.
        mode : {"hub", "local", "mlflow"}, default "local"
            The mode of the project.
        **kwargs : dict
            Extra keyword arguments passed to the project, depending on its mode.

            workspace : str or Path-like, optional
                See the :class:`Project` class docstring for details.

            tracking_uri : str, mode:mlflow only.
                The URI of the MLflow tracking server.
        """
        plugin, parameters = Project.__setup_plugin(mode, name, **kwargs)

        return plugin.delete(**parameters)

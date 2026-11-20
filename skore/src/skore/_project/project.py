"""Class definition of the ``skore`` project."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from pandas import DataFrame, Index, MultiIndex, RangeIndex

from skore._project import plugin
from skore._project.dependencies import assert_optional_dependencies_installed
from skore._project.summary import Summary
from skore._project.sync import synchronize
from skore._project.types import ProjectMode

if TYPE_CHECKING:
    from skore import CrossValidationReport, EstimatorReport
    from skore._project.experiments import Experiments


MODES = get_args(ProjectMode)


class Project:
    r"""
    API to manage a collection of key-report pairs.

    Its constructor initializes a project by creating a new project or by loading an
    existing one.

    The class main methods are :func:`~skore.Project.put`,
    :func:`~skore.Project.summarize`, :func:`~skore.Project.get`, and
    :func:`~skore.Project.sync`, respectively to insert a key-report pair into the
    project, obtain the metadata/metrics of the inserted reports, get a specific report
    by its id, and synchronize reports between projects.
    :func:`~skore.Project.delete` removes the project.

    Three mutually exclusive modes are available and can be configured using the
    ``mode`` parameter of the constructor:

    .. rubric:: Hub mode

    The project is configured to communicate with Skore Hub.

    In this mode, ``workspace`` is a Skore Hub concept that must be configured on
    the Skore Hub interface. It represents an isolated entity managing users,
    projects, and resources. It can be a company, organization, or team that operates
    independently within the system.

    Note: Using Project in ``hub`` mode requires an account on Skore Hub, with
    access rights to the specified workspace. Authentication uses an API key, in
    this order:

    - the ``SKORE_HUB_API_KEY`` environment variable, which always wins. The key
      is bound to a workspace; using it against another workspace fails at
      runtime.
    - otherwise the local credentials registry for the hub host and workspace:

    .. code-block:: bash

        $ skore hub api-key generate --host <host> --workspace <workspace>

    .. rubric:: Local mode

    Otherwise, the project is configured to the ``local`` mode to be persisted on
    the user machine in a skore ``workspace``, which is a directory named 'skore'
    containing a file named '.SKORE_WORKSPACE'.

    | The workspace can be set using kwargs or the environment variable
      ``SKORE_WORKSPACE``.
    | If not specified, it will be set to a default location chosen as follows:

        - we search the current working directory and its parents for a
          skore workspace. If we find one, we use it.
        - otherwise, if we are in a Git repository, we create a skore workspace at the
          root of the repository.
        - otherwise, we create a skore workspace in the current working directory.

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

        host : str, optional
            Hub address when ``mode="hub"`` (for example
            ``https://skore.probabl.ai``). If omitted,
            ``SKORE_HUB_URI`` is used. Ignored for other modes.

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
                | If not provided, a workspace is found or created according to those
                  rules (see find_workspace() in this module):

                      - if the SKORE_WORKSPACE environment variable is set, use that.
                      - otherwise look in the current working directory and its
                        parent for a skore workspace: a directory named
                        'skore' containing a file named
                        '.SKORE_WORKSPACE'. If found, use that.
                      - otherwise if we are in a Git repository, create
                        'skore' at the root of the repository.
                      - otherwise create 'skore' in the current working directory.

            host : str, optional
                Hub address when ``mode="hub"`` (for example
                ``https://skore.probabl.ai``). If omitted,
                ``SKORE_HUB_URI`` is used. Ignored for other modes.

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

        self.__mode: ProjectMode = mode
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

    @property
    def objective(self) -> str:
        """The project objective, as plain text."""
        return self.__project.objective()

    @objective.setter
    def objective(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"Objective must be a string (found '{type(text)}')")
        self.__project.set_objective(text)

    @property
    def experiments(self) -> Experiments:
        """The experiments stored in this project.

        Returns
        -------
        experiments : Experiments
            The collection of experiment rows. Do not construct this object
            directly.

        See Also
        --------
        Experiments :
            List and retrieve experiments.
        Experiment :
            One experiment and the methods that move it.
        """
        from skore._project.experiments import Experiments

        return Experiments(self)

    def _experiment_rows(self):
        experiments = getattr(self.__project, "experiments", None)
        if not callable(experiments):
            return []
        rows = experiments()
        return rows if isinstance(rows, list) else []

    def _write_experiment(self, row):
        """Validate one transition and persist that row."""
        from skore._project.experiments import (
            Experiment,
            normalize_row,
            transition_allowed,
        )

        stored = normalize_row(row)
        current = None
        for existing in self._experiment_rows():
            if str(existing["id"]) == stored["id"]:
                current = existing["status"]
                break
        if not transition_allowed(current, stored["status"], stored["plan"]):
            raise ValueError(
                f"Cannot move experiment {stored['id']} from {current!r} "
                f"to {stored['status']!r}."
            )
        self.__project.upsert_experiment(stored)
        return Experiment(self, stored)

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
        report : EstimatorReport or CrossValidationReport
            The same report, with an id assigned when it did not have one.

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

        stored = self._store_report(key, report)
        self._raise_check_ideas(key, stored)
        return stored

    def _store_report(self, key: str, report: EstimatorReport | CrossValidationReport):
        """Persist ``report`` without changing experiment rows."""
        from skore._project.experiments import ensure_report_id

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

        ensure_report_id(report)
        self.__project.put(key=key, report=report)
        return report

    def _raise_check_ideas(
        self, key: str, report: EstimatorReport | CrossValidationReport
    ):
        """Append one idea per issue or tip check, once per source and code."""
        from skore._project.experiments import normalize_row
        from skore.agent import create_experiment

        cache = getattr(report, "_check_results_cache", None) or {}
        if not cache:
            try:
                report.checks.summarize(fast_mode=True)
            except Exception:
                return
            cache = getattr(report, "_check_results_cache", None) or {}
        if report.id is None:
            return
        try:
            frame = self.experiments.frame
        except Exception:
            return
        for code, result in cache.items():
            section = result.get("section") if isinstance(result, dict) else None
            if section not in {"issue", "tip"}:
                continue
            title = result.get("title") if isinstance(result, dict) else None
            if not title:
                continue
            if frame.empty:
                matched = frame
            else:
                same_source = (
                    frame["source_id"].fillna("").astype(str).eq(str(report.id))
                )
                same_code = frame["check_code"].fillna("").astype(str).eq(str(code))
                matched = frame.loc[same_source & same_code]
            if not matched.empty:
                existing = normalize_row(matched.iloc[0].to_dict())
                if existing["status"] == "idea" and existing["question"] != title:
                    try:
                        self._write_experiment({**existing, "question": title})
                    except Exception:
                        return
                continue
            stem = f"{key}-{str(code).lower()}"
            suffix = 2
            taken = set() if frame.empty else set(frame["stem"].astype(str))
            while stem in taken:
                stem = f"{key}-{str(code).lower()}-{suffix}"
                suffix += 1
            try:
                create_experiment(
                    self,
                    stem=stem,
                    question=str(title),
                    source_id=str(report.id),
                    check_code=str(code),
                )
            except Exception:
                return
            frame = self.experiments.frame

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
        frame = DataFrame(sorted(records, key=lambda r: r["date"]), copy=False)
        if not frame.empty:
            frame.index = MultiIndex.from_arrays(
                [
                    RangeIndex(len(frame)),
                    Index(frame.pop("id"), name="id", dtype=str),
                ]
            )
        return Summary(frame, self)

    def sync(
        self,
        other: Project | ProjectMode,
        *,
        bidirectional: bool = False,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> DataFrame:
        """Copy missing reports to another project.

        Reports are matched using the ``report_id`` column returned by
        ``Project.summarize().frame()``. Set ``bidirectional=True`` to copy missing
        reports in both directions.

        Parameters
        ----------
        other : Project or {"hub", "local", "mlflow"}
            Destination project. When a mode is given, build the destination with this
            project's name and the mode-specific keyword arguments.
        bidirectional : bool, default=False
            If ``False``, transfer reports from this project to ``other``. If ``True``,
            also transfer reports missing from this project.
        dry_run : bool, default=False
            Return the planned operations without loading or storing reports.
        **kwargs : dict
            Mode-specific arguments used to build the destination when ``other`` is a
            mode string. For example, pass ``workspace`` for ``"hub"`` or
            ``tracking_uri`` for ``"mlflow"``.

        Returns
        -------
        result : pandas.DataFrame
            Synchronization status indexed by ``report_id``. The ``direction`` column
            is ``"outbound"`` from this project to ``other``, ``"inbound"`` from
            ``other`` to this project, or missing for skipped reports. The ``status``
            column is ``"planned"``, ``"transferred"``, or ``"skipped"``.
        """
        if isinstance(other, str):
            other = Project(self.name, mode=other, **kwargs)
        elif kwargs:
            raise TypeError(
                "Extra keyword arguments are only supported when `other` is a mode "
                "string."
            )
        elif not isinstance(other, Project):
            raise TypeError(f"`other` must be a Project (found {type(other)!r}).")
        if (
            self.mode == other.mode == "mlflow"
            and self.tracking_uri != other.tracking_uri
        ):
            # MLflow uses process-global tracking state. Crossing stores could
            # therefore read from or write to the wrong backend.
            raise ValueError(
                "Synchronization between MLflow projects requires the same "
                "tracking URI."
            )

        return synchronize(
            self,
            other,
            bidirectional=bidirectional,
            dry_run=dry_run,
        )

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

            host : str, optional
                Hub address when ``mode="hub"``. See the :class:`Project` class
                docstring for details.
        """
        plugin, parameters = Project.__setup_plugin(mode, name, **kwargs)

        return plugin.delete(**parameters)

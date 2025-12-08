"""Class definition of the ``skore`` project."""

from __future__ import annotations

import re
from importlib.metadata import entry_points
from typing import Any, Literal

from skore import CrossValidationReport, EstimatorReport
from skore._sklearn.types import MLTask
from skore.project.summary import Summary


class Project:
    r"""
    API to manage a collection of key-report pairs.

    Its constructor initializes a project by creating a new project or by loading an
    existing one.

    The class main methods are :func:`~skore.Project.put`,
    :func:`~skore.Project.summarize` and :func:`~skore.Project.get`, respectively to
    insert a key-report pair into the project, to obtain the metadata/metrics of the
    inserted reports and to get a specific report by its id.

    Two mutually exclusive modes are available and can be configured using the ``name``
    parameter of the constructor:

    .. rubric:: Hub mode

    If the ``name`` takes the form of the URI ``hub://<tenant>/<name>``, the project
    is configured to the ``hub`` mode to communicate with the ``skore hub``.

    A tenant is a ``skore hub`` concept that must be configured on the ``skore hub``
    interface. It represents an isolated entity managing users, projects, and
    resources. It can be a company, organization, or team that operates
    independently within the system.

    In this mode, you must have an account to the ``skore hub`` and must be
    authorized to the specified tenant. You must also be authenticated beforehand,
    using the ``skore-hub-login`` CLI.

    .. rubric:: Local mode

    Otherwise, the project is configured to the ``local`` mode to be persisted on
    the user machine in a directory called ``workspace``.

    | The workspace can be shared between all the projects.
    | The workspace can be set using kwargs or the environment variable
      ``SKORE_WORKSPACE``.
    | If not, it will be by default set to a ``skore/`` directory in the user
      cache directory:

    - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
    - on Linux, usually ``${HOME}/.local/share/skore``,
    - on macOS, usually ``${HOME}/Library/Application Support/skore``.

    Refer to the :ref:`project` section of the user guide for more details.

    Parameters
    ----------
    name : str
        The name of the project:

        - if the ``name`` takes the form of the URI ``hub://<tenant>/<name>``, the
          project is configured to communicate with the ``skore hub``,
        - otherwise, the project is configured to communicate with a local storage, on
          the user machine.
    **kwargs : dict
        Extra keyword arguments passed to the project, depending on its mode.

        workspace : Path, mode:local only.
            The directory where the local project is persisted.

    Attributes
    ----------
    name : str
        The name of the project, extrapolated from the ``name`` parameter.
    mode : str
        The mode of the project, extrapolated from the ``name`` parameter.
    ml_task : MLTask
        The ML task of the project; unset until a first report is put.

    Examples
    --------
    Construct reports.

    >>> from sklearn.datasets import make_classification, make_regression
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from skore import CrossValidationReport, EstimatorReport
    >>>
    >>> X, y = make_classification(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> classifier = LogisticRegression(max_iter=10)
    >>> classifier_report = EstimatorReport(
    >>>     classifier,
    >>>     X_train=X_train,
    >>>     y_train=y_train,
    >>>     X_test=X_test,
    >>>     y_test=y_test,
    >>> )
    >>>
    >>> X, y = make_regression(random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> regressor = LinearRegression()
    >>> regressor_report = EstimatorReport(
    >>>     regressor,
    >>>     X_train=X_train,
    >>>     y_train=y_train,
    >>>     X_test=X_test,
    >>>     y_test=y_test,
    >>> )
    >>>
    >>> cv_regressor_report = CrossValidationReport(regressor, X, y)

    Construct the project in local mode, persisted in a temporary directory.

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from skore import Project
    >>>
    >>> tmpdir = TemporaryDirectory().name
    >>> local_project = Project("my-xp", workspace=Path(tmpdir))

    Put reports in the project.

    >>> local_project.put("my-simple-regression", regressor_report)
    >>> local_project.put("my-simple-cv_regression", cv_regressor_report)

    Investigate metadata/metrics to filter the best reports.

    >>> summary = local_project.summarize()
    >>> summary = summary.query("rmse < 67")
    >>> reports = summary.reports()

    See Also
    --------
    :class:`~skore.project.summary.Summary` :
        DataFrame designed to investigate persisted reports' metadata/metrics.
    """

    __HUB_NAME_PATTERN = re.compile(r"hub://(?P<tenant>[^/]+)/(?P<name>.+)")

    @staticmethod
    def __setup_plugin(name: str) -> tuple[Literal["local", "hub"], str, Any, dict]:
        if not (PLUGINS := entry_points(group="skore.plugins.project")):
            raise SystemError("No project plugin found, please install at least one.")

        mode: Literal["local", "hub"]

        if match := re.match(Project.__HUB_NAME_PATTERN, name):
            mode = "hub"
            name = match["name"]
            parameters = {"tenant": match["tenant"], "name": name}
        else:
            mode = "local"
            parameters = {"name": name}

        if mode not in PLUGINS.names:
            raise ValueError(f"Unknown mode `{mode}`. Please install `skore[{mode}]`.")

        return mode, name, PLUGINS[mode].load(), parameters

    def __init__(self, name: str, **kwargs):
        r"""
        Initialize a project.

        Parameters
        ----------
        name : str
            The name of the project:

            - if the ``name`` takes the form of the URI ``hub://<tenant>/<name>``, the
              project is configured to communicate with the ``skore hub``,
            - otherwise, the project is configured to communicate with a local storage,
              on the user machine.
        **kwargs : dict
            Extra keyword arguments passed to the project, depending on its mode.

            workspace : Path, mode:local only.
                The directory where the local project is persisted.

                | The workspace can be shared between all the projects.
                | The workspace can be set using kwargs or the environment variable
                  ``SKORE_WORKSPACE``.
                | If not, it will be by default set to a ``skore/`` directory in the
                  USER cache directory:

                - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
                - on Linux, usually ``${HOME}/.cache/skore``,
                - on macOS, usually ``${HOME}/Library/Caches/skore``.
        """
        mode, name, plugin, parameters = Project.__setup_plugin(name)

        self.__mode = mode
        self.__name = name
        self.__project = plugin(**(kwargs | parameters))

        ml_tasks = {report["ml_task"] for report in self.__project.summarize()}

        if len(ml_tasks) > 1:
            raise RuntimeError(
                "Expected every report in the Project to have the same ML task. "
                f"Got ML tasks {ml_tasks}."
            )

        self.ml_task: MLTask | None = ml_tasks.pop() if ml_tasks else None

    @property
    def mode(self):
        """The mode of the project."""
        return self.__mode

    @property
    def name(self):
        """The name of the project."""
        return self.__name

    def put(self, key: str, report: EstimatorReport | CrossValidationReport):
        """
        Put a key-report pair to the project.

        If the key already exists, its last report is modified to point to this new
        report, while keeping track of the report history.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the project.
        report : EstimatorReport | CrossValidationReport
            The report to associate with ``key`` in the project.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
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
                    f"Expected a report meant for ML task {self.ml_task!r} "
                    f"but the given report is for ML task {report.ml_task!r}"
                )
        else:
            self.ml_task = report.ml_task

        return self.__project.put(key=key, report=report)

    def get(self, id: str) -> EstimatorReport | CrossValidationReport:
        """
        Get a persisted report by its id.

        Report IDs can be found via :meth:`skore.Project.summarize`, which is also the
        preferred method of interacting with a ``skore.Project``.

        Parameters
        ----------
        id : str
            The id of a report already put in the ``project``.

        Raises
        ------
        KeyError
            If a non-existent ID is passed.
        """
        return self.__project.get(id)

    def summarize(self) -> Summary:
        """Obtain metadata/metrics for all persisted reports."""
        return Summary.factory(self.__project)

    def __repr__(self) -> str:  # noqa: D105
        return self.__project.__repr__()

    @staticmethod
    def delete(name: str, **kwargs):
        r"""
        Delete a project.

        Parameters
        ----------
        name : str
            The name of the project:

            - if the ``name`` takes the form of the URI ``hub://<tenant>/<name>``, the
              project is configured to communicate with the ``skore hub``,
            - otherwise, the project is configured to communicate with a local storage,
              on the user machine.
        **kwargs : dict
            Extra keyword arguments passed to the project, depending on its mode.

            workspace : Path, mode:local only.
                The directory where the local project is persisted.

                | The workspace can be shared between all the projects.
                | The workspace can be set using kwargs or the environment variable
                  ``SKORE_WORKSPACE``.
                | If not, it will be by default set to a ``skore/`` directory in the
                  USER cache directory:

                - on Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
                - on Linux, usually ``${HOME}/.cache/skore``,
                - on macOS, usually ``${HOME}/Library/Caches/skore``.
        """
        _, _, plugin, parameters = Project.__setup_plugin(name)

        return plugin.delete(**(kwargs | parameters))

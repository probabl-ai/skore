"""Class definition of the ``skore`` project."""

from __future__ import annotations

import re
import sys
import types

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from ..sklearn._estimator.report import EstimatorReport
from .metadata import Metadata


class Project:
    r"""
    API to manage a collection of key-report pairs.

    Its constructor initializes a project by creating a new project or by loading an
    existing one.

    The class main methods are :func:`~skore.Project.put`,
    :func:`~skore.Project.reports.metadata` and :func:`~skore.Project.reports.get`,
    respectively to insert a key-report pair into the project, to obtain the metadata of
    the inserted reports and to get a specific report by its id.

    Two mutually exclusive modes are available and can be configured using the ``name``
    parameter of the constructor:

    - mode : hub

        If the ``name`` takes the form of the URI ``hub://<tenant>/<name>``, the project
        is configured to the ``hub`` mode to communicate with the ``skore hub``.

        A tenant is a ``skore hub`` concept that must be configured on the ``skore hub``
        interface. It represents an isolated entity managing users, projects, and
        resources. It can be a company, organization, or team that operates
        independently within the system.

        In this mode, you must have an account to the ``skore hub`` and must be
        authorized to the specified tenant. You must also be authenticated beforehand,
        using the ``skore-hub-login`` CLI.

    - mode : local

        Otherwise, the project is configured to the ``local`` mode to be persisted on
        the user machine in a directory called ``workspace``.

        The workspace can be shared between all the projects.
        The workspace can be set using kwargs or the envar ``SKORE_WORKSPACE``.
        If not, it will be by default set to a ``skore/`` directory in the USER
        cache directory:

        - in Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
        - in Linux, usually ``${HOME}/.cache/skore``,
        - in macOS, usually ``${HOME}/Library/Caches/skore``.

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

            The workspace can be shared between all the projects.
            The workspace can be set using kwargs or the envar ``SKORE_WORKSPACE``.
            If not, it will be by default set to a ``skore/`` directory in the USER
            cache directory:

            - in Windows, usually ``C:\Users\%USER%\AppData\Local\skore``,
            - in Linux, usually ``${HOME}/.cache/skore``,
            - in macOS, usually ``${HOME}/Library/Caches/skore``.

    Attributes
    ----------
    name : str
        The name of the project, extrapolated from the ``name`` parameter.
    mode : str
        The mode of the project, extrapolated from the ``name`` parameter.

    Examples
    --------
    Construct reports.

    >>> from sklearn.datasets import make_classification, make_regression
    >>> from sklearn.linear_model import LinearRegression, LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from skore.sklearn import EstimatorReport
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

    Construct the project in local mode, persisted in a temporary directory.

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from skore import Project
    >>>
    >>> tmpdir = TemporaryDirectory().name
    >>> local_project = Project("my-xp", workspace=Path(tmpdir))

    Put reports in the project.

    >>> local_project.put("my-simple-classification", classifier_report)
    >>> local_project.put("my-simple-regression", regressor_report)

    Investigate metadata/metrics to filter the best reports.

    >>> metadata = local_project.reports.metadata()
    >>> metadata = metadata.query("ml_task.str.contains('regression') and (rmse < 67)")
    >>> reports = metadata.reports()

    See Also
    --------
    Metadata : DataFrame designed to investigate persisted reports' metadata/metrics.
    """

    __HUB_NAME_PATTERN = re.compile(r"hub://(?P<tenant>[^/]+)/(?P<name>.+)")

    def __init__(self, name: str, **kwargs):
        if not (PLUGINS := entry_points(group="skore.plugins.project")):
            raise SystemError("No project plugin found, please install at least one.")

        if match := re.match(self.__HUB_NAME_PATTERN, name):
            mode = "hub"
            name = match["name"]
            kwargs |= {"tenant": match["tenant"], "name": name}
        else:
            mode = "local"
            kwargs |= {"name": name}

        if mode not in PLUGINS.names:
            raise ValueError(
                f"Unknown mode `{mode}`. "
                f"Please install the `skore-{mode}-project` python package."
            )

        self.__mode = mode
        self.__name = name
        self.__project = PLUGINS[mode].load()(**kwargs)

    @property
    def mode(self):
        """The mode of the project."""
        return self.__mode

    @property
    def name(self):
        """The name of the project."""
        return self.__name

    def put(self, key: str, report: EstimatorReport):
        """
        Put a key-report pair to the project.

        If the key already exists, its last report is modified to point to this new
        report, while keeping track of the report history.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the project.
        report : skore.EstimatorReport
            The report to associate with ``key`` in the project.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        if not isinstance(report, EstimatorReport):
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` (found '{type(report)}')"
            )

        return self.__project.put(key=key, report=report)

    @property
    def reports(self):
        """Accessor for interaction with the persisted reports."""

        def get(id: str) -> EstimatorReport:  # hide underlying functions from user
            """Get a persisted report by its id."""
            return self.__project.reports.get(id)

        def metadata() -> Metadata:  # hide underlying functions from user
            """Obtain metadata/metrics for all persisted reports."""
            return Metadata.factory(self.__project)

        return types.SimpleNamespace(get=get, metadata=metadata)

    def __repr__(self) -> str:  # noqa: D105
        return self.__project.__repr__()

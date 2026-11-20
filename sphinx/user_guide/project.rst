.. _project:

==============================
Storing data science artifacts
==============================

.. currentmodule:: skore

`skore` provides a :class:`Project` class to store data science artifacts. The storage
is either local or remote, based on the value passed to the parameter `mode` at
initialization. When `mode` is set to `hub`, the project is configured to communicate
with Skore Hub. Authentication uses an API key, in this order:

- the ``SKORE_HUB_API_KEY`` environment variable, which always wins. The key
  is bound to a workspace; using it against another workspace fails at
  runtime.
- otherwise the local credentials registry for the hub host and workspace:

.. code-block:: bash

    $ skore hub api-key generate --host <host> --workspace <workspace>

See :ref:`cli-hub` to generate, list, and revoke keys from the command line.
Refer to the documentation of :class:`Project` for the detailed API.

Creating a project
------------------

All modes share the same constructor shape: pass ``name``, ``mode``, and any
mode-specific keyword arguments.

.. code-block:: python

   from pathlib import Path
   from skore import Project

   # Local persistence
   project_local = Project(name="my-xp", mode="local", workspace=Path("/tmp/skore"))

   # Skore Hub (requires an API key in the registry or SKORE_HUB_API_KEY)
   project_hub = Project(
       name="my-xp",
       mode="hub",
       workspace="my-workspace",
       host="my-host",
   )

   # MLflow experiment
   project_mlflow = Project(
       name="my-experiment",
       mode="mlflow",
       tracking_uri="http://localhost:5000",
   )

.. note::

   Databricks managed MLflow (``tracking_uri="databricks"``) only accepts experiment
   names that are absolute workspace paths, so ``name`` must be one, for example
   ``"/Users/me@example.com/my-experiment"``. `skore` places the internal experiment in
   which it stores report objects in the home directory of the workspace user.

Working with reports
--------------------

Once a project is created, store an :class:`EstimatorReport` or
:class:`CrossValidationReport` via :meth:`Project.put`.

To retrieve the reports stored in the project, use the project summary by calling the
method :meth:`Project.summarize`. This method returns a ``Summary`` object that holds
the metadata and metrics of the stored reports and renders as an interactive table in
Jupyter-like environments. Reports are listed in ascending order of their ``date``.

The interactive view provides different views to sort, group by, and filter the reports;
the selection produces a query string ready to pass to ``Summary.query(...)``. Once the
reports are filtered, retrieve them by calling the ``compare`` method on the object
returned by :meth:`Project.summarize`. This method returns a list of
:class:`EstimatorReport` or :class:`CrossValidationReport` instances (or a
:class:`ComparisonReport` when called with ``return_as="report"``).

To retrieve a specific report for which you have its ``id`` (as returned by
:meth:`Project.summarize`), use the :meth:`Project.get` method.
:meth:`Project.delete` removes the project itself.

Working with experiments
------------------------

:attr:`Project.experiments` returns an :class:`Experiments` collection. Its
:attr:`Experiments.frame` attribute is a :class:`pandas.DataFrame` with one row
per experiment. Retrieve one row with :meth:`Experiments.get`.

Do not construct these objects yourself. :func:`skore.agent.create_experiment`
appends a row and returns an :class:`Experiment`. Omitting ``plan`` stores an
idea. Passing ``plan`` stores a backlog row.

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.dummy import DummyRegressor

   from skore.agent import create_experiment

   X, y = make_regression(n_samples=20, n_features=3, random_state=0)
   idea = create_experiment(
       project_local,
       stem="01_baseline",
       question="What RMSE does a constant predictor reach?",
   )
   experiment = idea.promote("DummyRegressor, 2-fold cross-validation.")
   report = experiment.run(DummyRegressor(), X, y, splitter=2)

:meth:`Experiment.decide` moves a row among ``idea``, ``aside``, and
``discarded``. :meth:`Experiment.promote` moves an idea to the backlog.
:meth:`Experiment.run` evaluates an estimator once and stores the report under
the experiment id. A second run is rejected.

.. _synchronizing-projects:

Synchronizing projects
----------------------

Use :meth:`Project.sync` to transfer reports between projects. The project on which the
method is called is the source. :ref:`cli-sync` runs the same transfer from the shell.

.. code-block:: python

   project_hub = Project(
       name=project_local.name,
       mode="hub",
       workspace="my-workspace",
       host="my-host",
   )
   result = project_local.sync(project_hub)

The caller is the source; reverse the call for the opposite direction. Set
``bidirectional=True`` to transfer missing reports in both directions.

When both projects have the same name, pass the destination mode as a shortcut. The
destination is built with the caller's name and the supplied mode-specific arguments.

.. code-block:: python

   result = project_local.sync("hub", workspace="my-workspace", host="my-host")

Reports are matched using the ``report_id`` column returned by
``Project.summarize().frame()`` and copied with their keys. Existing IDs are skipped;
contents and metadata are not compared. Reports without a ``report_id`` are ignored.
Set ``dry_run=True`` to return the transfer plan without loading or storing reports.

The returned :class:`pandas.DataFrame` is indexed by ``report_id``. Its ``direction``
column is ``"outbound"`` from the caller to the other project, ``"inbound"`` from the
other project to the caller, or missing when a report is skipped. Its ``status`` column
is ``"planned"``, ``"transferred"``, or ``"skipped"``.

.. note::

   Two MLflow projects must use the same tracking URI because MLflow uses process-global
   tracking state. Synchronization does not provide concurrency control.

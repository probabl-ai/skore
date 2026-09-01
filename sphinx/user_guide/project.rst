.. _project:

==============================
Storing data science artifacts
==============================

.. currentmodule:: skore

`skore` provides a :class:`Project` class to store data science artifacts. The storage
is either local or remote, based on the value passed to the parameter `mode` at
initialization. When `mode` is set to `hub`, the project is configured to communicate
with `skore hub`. Refer to the documentation of :class:`Project` for the detailed API
and take a look on the `example <example-getting-started_>`_.

Creating a project
------------------

All modes share the same constructor shape: pass ``name``, ``mode``, and any
mode-specific keyword arguments.

.. code-block:: python

   from pathlib import Path
   from skore import Project

   # Local persistence
   project_local = Project(name="my-xp", mode="local", workspace=Path("/tmp/skore"))

   # Skore Hub (requires skore.login() first)
   project_hub = Project(name="my-xp", mode="hub", workspace="my-workspace")

   # MLflow experiment
   project_mlflow = Project(
       name="my-experiment",
       mode="mlflow",
       tracking_uri="http://localhost:5000",
   )

Working with reports
--------------------

Once a project is created, store :class:`EstimatorReport` via the method
:meth:`Project.put`.

To retrieve the reports stored in the project, use the project summary by calling the
method :meth:`Project.summarize`. This method returns a ``Summary`` object that holds
the metadata and metrics of the stored reports and renders as an interactive table in
Jupyter-like environments. Reports are listed in ascending order of their ``date``.

The interactive view provides different views to sort, group by, and filter the reports;
the selection produces a query string ready to pass to ``Summary.query(...)``. Once the
reports are filtered, retrieve them by calling the ``compare`` method on the object
returned by :meth:`Project.summarize`. This method returns a list of
:class:`EstimatorReport` instances (or a :class:`ComparisonReport` when called with
``return_as="report"``).

To retrieve a specific report for which you have its ``id`` (as returned by
:meth:`Project.summarize`), use the :meth:`Project.get` method.

Synchronizing projects
----------------------

Use :meth:`Project.sync` to transfer reports between projects. The project on which the
method is called is the source.

.. code-block:: python

   project_hub = Project(
       name=project_local.name,
       mode="hub",
       workspace="my-workspace",
   )
   result = project_local.sync(project_hub)

The caller is the source; reverse the call for the opposite direction. Set
``bidirectional=True`` to transfer missing reports in both directions.

When both projects have the same name, pass the destination mode as a shortcut. The
destination is built with the caller's name and the supplied mode-specific arguments.

.. code-block:: python

   result = project_local.sync("hub", workspace="my-workspace")

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

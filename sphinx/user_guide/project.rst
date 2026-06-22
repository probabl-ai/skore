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
   project_hub = Project(name="my-xp", workspace="my-workspace", mode="hub")

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
:meth:`Project.summarize`), use the method :meth:`Project.get` to retrieve the
:class:`EstimatorReport`.

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

Once a project is created, store :class:`EstimatorReport` via the method
:meth:`Project.put`.

To retrieve the reports stored in the project, use the project summary by calling the
method :meth:`Project.summarize`. This method returns a ``Summary`` object that holds
the metadata and metrics of the stored reports and renders as an interactive table in
Jupyter-like environments.

The interactive view provides different views to sort, group by, and filter the reports;
the selection produces a query string ready to pass to ``Summary.query(...)``. Once the
reports are filtered, retrieve them by calling the ``compare`` method on the object
returned by :meth:`Project.summarize`. This method returns a list of
:class:`EstimatorReport` instances (or a :class:`ComparisonReport` when called with
``return_as="report"``).

To retrieve a specific report for which you have its `id`, use the method
:meth:`Project.get` to retrieve the :class:`EstimatorReport`.

Synchronizing across modes
--------------------------

Projects in different modes (``local``, ``hub``, ``mlflow``) can be reconciled with
:meth:`Project.sync_with`. This is useful when you work offline in local mode and upload
experiments once connectivity is restored:

.. code-block:: python

    from skore import Project, login

    local = Project("my-xp", mode="local")
    local.put("baseline", report)

    login(mode="hub")
    hub = Project("my-workspace/my-xp", mode="hub")

    result = local.sync_with(hub, direction="put")
    print(result.summary())

Use ``direction="get"`` to download remote reports locally, or ``direction="both"`` for
bidirectional reconciliation. When the same key refers to different reports on both
sides, control the outcome with ``on_conflict`` (for example ``"latest_wins"`` or
``"skip"``). See :class:`SyncResult` for the outcome details.

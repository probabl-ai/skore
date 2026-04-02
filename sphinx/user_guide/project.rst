.. _project:

===================
Manage your reports
===================

.. currentmodule:: skore

When you evaluate models with Skore, you get rich **reports**—structured objects that
capture metrics, diagnostics, and context. A :class:`Project` is where those reports
**live** between sessions: a named collection you can grow over time, compare, and load
back into Python.

You pick the storage backend with the ``mode`` argument when you construct the project
(for example ``mode="local"``, ``mode="hub"``, or ``mode="mlflow"``). The sections below
first cover **storing** (where data goes and how :meth:`Project.put` behaves), then
**retrieving** (overview with :meth:`Project.summarize` and single-report access with
:meth:`Project.get`).

Storing reports
---------------

**Storing** combines two ideas: **where** reports are persisted (local disk, Skore Hub,
or MLflow), and **how** you write them into the project. The three modes differ in
privacy, collaboration, and integration with other tooling, but the entry point is
always :meth:`Project.put`.

Regardless of mode, a project may only contain reports for **one ML task** (for example
all regression or all classification) so comparisons stay meaningful. Skore enforces
that once you start storing reports.

To save a report, call ``put`` with a string **key** (a name you choose for that run
within the project) and an :class:`EstimatorReport` or :class:`CrossValidationReport`.
If you call ``put`` again with the same key, the project keeps **history**, but the
key’s **current** report becomes the new one.

Local mode: your machine, your copy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Local** mode is the default. Everything stays on disk on the machine where you run
the code. You **do not start any server**—no tracking server, hub endpoint, or
background process—and nothing is sent over the network. It suits everyday work,
teaching, prototypes, and any case where your machine is the source of truth.

Several named projects can share one **workspace** directory; you can set it explicitly
or rely on a default location for your OS. When you ``put``, Skore writes into that
local layout; the **key** is simply the human-readable label for the run inside the
project. See :ref:`example_skore_local_project` for a full script.

Hub mode: shared, remote projects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Hub** mode targets teams that want a **central** place for reports: access control, a
shared namespace, and storage on **Skore Hub** instead of only on users' laptops. Use it
when others must see the same projects or when hosting and governance matter.

You authenticate (for example via the client login flow) and name the project with a
**workspace** and **project** as configured in the Skore Hub UI. Each ``put`` sends the
report to that remote project; the **key** is again your label for the run within the
project. See the `Hub Skore Project <example-skore-hub-project_>`_ example.

MLflow mode: meet your existing tracking stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**MLflow** mode fits stacks that already use **MLflow**—shared tracking servers,
experiment views, or pipelines that expect runs, metrics, and artifacts there. Skore
**complements** MLflow: your reports show up as runs in an MLflow **experiment** (the
project ``name`` is the experiment name), next to other logged work.

Here ``put`` has extra meaning: the **key** becomes the **MLflow run name**, and each
``put`` creates a **new run**, logs metrics and related artifacts, and stores a pickled
copy of the report so it can be reloaded. You must not already have another MLflow run
open in the same process when you call ``put``. See :ref:`example_skore_mlflow_project`
for a runnable setup (including a local MLflow server).

Retrieving reports
------------------

Summary overview
^^^^^^^^^^^^^^^^

Call :meth:`Project.summarize` to see **every** report currently in the project at once.
The return value is a **Summary**: a table of metadata and metrics (one row per report)
built on :class:`pandas.DataFrame`, so you can sort, slice, and use
:meth:`pandas.DataFrame.query` like any dataframe.

In Jupyter, with the optional widget extras installed, the summary can also render as an
interactive view (often with a parallel-coordinates plot) to explore and filter runs.
Whether you use the widget or the plain table, it is the same object.

After filtering if you wish, call ``.reports()`` on that summary to load the matching
full :class:`EstimatorReport` or :class:`CrossValidationReport` instances—typical when
you want to compare several candidates or open a short list of finalists.

Single report by ID
^^^^^^^^^^^^^^^^^^^

When you already know which stored report you need, :meth:`Project.get` loads it by
**ID**. IDs show up in the summary index (and in the MLflow UI for MLflow-backed
projects). They depend on ``mode``:

- **Local** and **hub**: Skore’s report ID for that stored object.
- **MLflow**: the **MLflow run ID** of the run created by ``put``.

If nothing matches, ``get`` raises ``KeyError``. For full signatures and edge cases, see
:class:`Project` and the `getting started example <example-getting-started_>`_.

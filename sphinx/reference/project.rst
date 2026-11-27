Managing a project
------------------

.. currentmodule:: skore

Skore project
^^^^^^^^^^^^^

These functions and classes are meant for managing a `Project` and its reports.

.. autosummary::
   :toctree: api/
   :template: class_with_accessors.rst

   Project

.. rubric:: Methods

.. autosummary::
   :toctree: api/
   :template: class_methods_no_index.rst

   Project.put
   Project.get
   Project.summarize
   Project.sync
   Project.experiments
   Project.delete

Skore project's summary
^^^^^^^^^^^^^^^^^^^^^^^

When calling :meth:`Project.summarize`, returns a :class:`Summary` object that
holds the metadata and metrics of the stored reports as a
:class:`pandas.DataFrame` (accessible via its :meth:`~Summary.frame` method) and
renders an interactive table in Jupyter-like environments to filter and retrieve
the reports.

The returned object is not intended to be instantiated directly. Always use
:meth:`Project.summarize`.

.. autosummary::
   :toctree: api/
   :template: class_with_accessors.rst

   Summary

.. rubric:: Methods

.. autosummary::
   :toctree: api/
   :template: class_methods_no_index.rst

   Summary.frame
   Summary.query
   Summary.compare

Experiments
^^^^^^^^^^^

An experiment is one unit of work on a project: an idea, a backlog item, or a
finished run. :func:`agent.create_experiment` appends a unit and returns a
handle. Writes on that handle send the version it was read from. A write based
on a stale version raises :class:`JournalConflictError`, whose ``current``
attribute is the unit as stored now.

.. autosummary::
   :toctree: api/
   :template: class_with_accessors.rst

   Experiment
   Experiments

.. rubric:: Methods

.. autosummary::
   :toctree: api/
   :template: class_methods_no_index.rst

   Experiment.decide
   Experiment.promote
   Experiment.update_plan
   Experiment.start
   Experiment.release
   Experiment.abandon
   Experiment.run
   Experiments.frame
   Experiments.get

.. autosummary::
   :toctree: api/
   :template: base.rst

   agent.create_experiment
   JournalConflictError

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

Skore project's summary
^^^^^^^^^^^^^^^^^^^^^^^

When calling :meth:`Project.summarize`, a :class:`~skore.project.summary.Summary` object
is returned. This object is a :class:`pandas.DataFrame` with a specific HTML
representation to allow you filter and retrieve the reports.

.. autosummary::
   :toctree: api/
   :template: class_without_inherited_members.rst

   project.summary.Summary

.. autosummary::
   :toctree: api/
   :template: class_methods_no_index.rst

   project.summary.Summary.reports

.. note::
   This class :class:`~skore.project.summary.Summary` is not meant to be used
   directly. Instead, use the method :meth:`Project.summarize`.

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

When calling :meth:`Project.summarize`, returns a summary table as a
:class:`pandas.DataFrame` with a specific HTML representation to allow
you to filter and retrieve the reports.

The returned object is not intended to be instantiated or imported directly.
Always use :meth:`Project.summarize`.

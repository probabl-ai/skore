Managing a project
------------------

.. currentmodule:: skore

Skore project
^^^^^^^^^^^^^

These functions and classes are meant for managing a `Project` and its reports.

.. autosummary::
    :toctree: api/
    :template: base.rst

    login

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

When calling :meth:`Project.summarize`, returns a ``Summary`` object that holds
the metadata and metrics of the stored reports as a :class:`pandas.DataFrame`
(accessible via its ``frame`` method) and renders an interactive table in
Jupyter-like environments to filter and retrieve the reports.

The returned object is not intended to be instantiated or imported directly.
Always use :meth:`Project.summarize`.

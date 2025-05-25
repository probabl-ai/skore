Managing a project
------------------

.. currentmodule:: skore

Skore project
^^^^^^^^^^^^^

These functions and classes are meant for managing a `Project` and its reports.

.. autosummary::
    :toctree: api/
    :template: base.rst

    Project

.. rubric:: Methods

.. autosummary::
   :toctree: ../api/
   :template: class_methods_no_index.rst

    Project.put

.. rubric:: Reports

.. autosummary::
   :toctree: ../api/
   :template: autosummary/accessor.rst

   Project.reports

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

   Project.reports.get
   Project.reports.metadata

Skore project's metadata
^^^^^^^^^^^^^^^^^^^^^^^^

When calling :meth:`Project.reports.metadata`, a :class:`Metadata` object is returned.
This object is a :class:`pandas.DataFrame` with a specific HTML representation to
allow you filter and retrieve the reports.

.. autosummary::
   :toctree: api/
   :template: class_without_inherited_members.rst

   project.metadata.Metadata

.. autosummary::
   :toctree: api/
   :template: class_methods_no_index.rst

   project.metadata.Metadata.reports

.. note::
   This class :class:`~skore.project.metadata.Metadata` is not meant to be used
   directly. Instead, use the accessor :meth:`Project.reports.metadata`.

.. _project:

==============================
Storing data science artifacts
==============================

.. currentmodule:: skore

`skore` provides a :class:`Project` class to store data science artifacts. The storage
is either local or remote, based on the value passed to the parameter `mode` at the
initialization. When `mode` is set to `hub`, the project is configured to communicate
with the `skore hub`. Refer to the documentation of :class:`Project` for the detailed
API.

Once a project is created, store :class:`EstimatorReport` via the method
:meth:`Project.put`.

To retrieve the reports stored in the project, use the project summary by calling the
method :meth:`Project.summarize`. This method returns a summary table with a rich HTML
representation in interactive Jupyter-like environments.

The summary view allows filtering reports using a parallel coordinates plot. Once the
reports are filtered, retrieve them by calling the ``reports`` method on the object
returned by :meth:`Project.summarize`. This method returns a list of
:class:`EstimatorReport` instances.

To retrieve a specific report for which you have its `id`, use the method
:meth:`Project.get` to retrieve the :class:`EstimatorReport`.

.. _project:

==============================
Storing data science artifacts
==============================

.. currentmodule:: skore

`skore` provides a :class:`Project` class to store data science artifacts. The storage
is either local or remote, based on the value passed to the parameter `name` at the
initialization. When `name` is set to the form of the URI `hub://<workspace>/<name>`,
the project is configured to the `hub` mode to communicate with the `skore hub`.
Refer to the documentation of :class:`Project` for the detailed API.

Once a project is created, store :class:`EstimatorReport` via the method
:meth:`Project.put`.

To retrieve the reports stored in the project, use the project summary by calling the
method :meth:`Project.summarize`. This method returns a
:class:`~skore.project.summary.Summary` instance that presents a rich HTML
representation in interactive Jupyter-like environments. A parallel coordinates plot is
shown to filter the reports based on different criteria. Once the reports are filtered
on the parallel coordinates plot, retrieve the reports by calling the method
:meth:`~skore.project.summary.Summary.reports` from the
:class:`~skore.project.summary.Summary` instance. This method returns a list of
:class:`EstimatorReport` instances.

To retrieve a specific report for which you have its `id`, use the method
:meth:`Project.get` to retrieve the :class:`EstimatorReport`.

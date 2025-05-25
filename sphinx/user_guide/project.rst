.. _project:

==============================
Storing data science artifacts
==============================

.. currentmodule:: skore

`skore` provides a :class:`Project` class to store data science artifacts. The storage
is either local or remote, based on the value passed to the parameter `name` at the
initialization. When `name` is set to the form of the URI `hub://<tenant>/<name>`,
the project is configured to the `hub` mode to communicate with the `skore hub`.
Refer to the documentation of :class:`Project` for the detailed API.

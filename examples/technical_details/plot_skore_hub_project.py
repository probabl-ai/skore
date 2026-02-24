"""
.. _example_skore_hub_project:

=================
Hub skore Project
=================

This example shows how to use :class:`~skore.Project` in **hub** mode: store
reports remotely and inspect them. A key point is that
:meth:`~skore.Project.summarize` returns a :class:`~skore.project._summary.Summary`,
which is a :class:`pandas.DataFrame`. In Jupyter you get an interactive widget, but
you can always inspect and filter the summary as a DataFrame if you prefer.

Examples
--------

Basic usage:

.. code-block:: bash

    WORKSPACE=<workspace> PROJECT=<project> python plot_skore_hub_project.py
"""

# %%
# .. testsetup::
from logging import getLogger
from os import environ
from sys import exit

logger = getLogger(__name__)

if environ.get("SPHINX_BUILD"):
    GITHUB = environ.get("GITHUB_ACTIONS")
    API_KEY = environ.get("SPHINX_EXAMPLE_API_KEY")
    WORKSPACE = environ.get("SPHINX_EXAMPLE_WORKSPACE")
    PROJECT = environ.get("SPHINX_EXAMPLE_PROJECT")

    if not (GITHUB and API_KEY and WORKSPACE and PROJECT):
        logger.warning("Example `_example_skore_hub_project` skipped.")
        exit(0)

    environ["SKORE_HUB_API_KEY"] = API_KEY
else:
    assert (WORKSPACE := environ.get("WORKSPACE")), "`WORKSPACE` must be defined."
    assert (PROJECT := environ.get("PROJECT")), "`PROJECT` must be defined."

# %%
from skore import login

login(mode="hub")

# %%
from skore import Project

Project.delete(f"{WORKSPACE}/{PROJECT}", mode="hub")

# %%
from skore import Project

Project(f"{WORKSPACE}/{PROJECT}", mode="hub")

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
"""

# %%
# .. testsetup::
from os import environ
from sys import exit

from skore import Project

if ("SPHINX_BUILD" in environ) and ("SKORE_HUB_API_KEY" not in environ):
    exit(0)

WORKSPACE = environ["SPHINX_EXAMPLE_WORKSPACE"]
PROJECT = environ["SPHINX_EXAMPLE_PROJECT"]

Project.delete(f"{WORKSPACE}/{PROJECT}", mode="hub")

# %%
from skore import login

login(mode="hub")

# %%
from skore import Project

project = Project(f"{WORKSPACE}/{PROJECT}", mode="hub")

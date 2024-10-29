"""
.. _example_basic_ml_usage:

==========================
Getting started with skore
==========================

This example illustrates the motivation and the use of
:func:`~skore.cross_validate` to get assistance when developing your
ML/DS projects.
"""


# %%
import subprocess

# remove the skore project if it already exists
subprocess.run("rm -rf my_project_gs_ml.skore".split())

# create the skore project
subprocess.run("python3 -m skore create my_project_gs_ml".split())

"""
==============================
Getting started with ``skore``
==============================

*This example is work in progress!*

This guide provides a quick start to ``skore``, an open-source package that aims to enable data scientists to:

#. Store objects of different types from their Python code: python lists, ``scikit-learn`` fitted pipelines, ``plotly`` figures, and more.
#. Track and visualize these stored objects on a user-friendly dashboard.
#. Export the dashboard to a HTML file.


Initialize a Project and launch the UI
--------------------------------------

From your shell, initialize a skore project, here named ``project``, that will be
in your current working directory:

.. code:: console

    python -m skore create "project"

This will create a ``skore`` project directory named ``project`` in the current
directory.

From your shell (in the same directory), start the UI locally:

.. code:: console

    python -m skore launch "project"

This will automatically open a browser at the UI's location.

Now that the project file exists, we can load it in our notebook so that we can
read from and write to it:

.. code-block:: python

    from skore import load

    project = load("project.skore")

Storing some items
------------------

This will automatically open a browser at the UI's location:

**CONTINUE WORKING HERE**

#. On the top left, create a new ``View``.
#. From the ``Elements`` section on the bottom left, you can add stored items to this view, either by double-cliking on them or by doing drag-and-drop.

.. image:: https://raw.githubusercontent.com/sylvaincom/sylvaincom.github.io/master/files/probabl/skore/2024_10_14_skore_demo.gif
   :alt: Getting started with ``skore`` demo
"""

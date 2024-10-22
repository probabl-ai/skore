.. _getting_started:

Getting started
===============

``skore`` UI
------------

.. currentmodule:: skore

From your shell, initialize a skore project, here named ``project.skore``, that
will be in your current working directory:

.. code:: console

    python -m skore create "project.skore"

Then, from your Python code (in the same directory), load the project and store
an integer for example:

.. code-block:: python

    from skore import load
    project = load("project.skore")
    project.put("my_int", 3)

Finally, from your shell (in the same directory), start the UI locally:

.. code:: console

    python -m skore launch project.skore

This will automatically open a browser at the UI's location:

#. On the top left, create a new ``View``.
#. From the ``Elements`` section on the bottom left, you can add stored items to this view, either by double-cliking on them or by doing drag-and-drop.

For more features, please look into :ref:`auto_examples`.

.. image:: https://raw.githubusercontent.com/sylvaincom/sylvaincom.github.io/master/files/probabl/skore/2024_10_14_skore_demo.gif
   :alt: Getting started with ``skore`` demo

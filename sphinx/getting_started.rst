.. _getting_started:

Getting started
===============

.. currentmodule:: skore

From your shell, initialize a skore project, here named ``my_project``:

.. code:: console

    python -m skore create "my_project"

This will create a skore project directory named ``my_project.skore`` in your
current working directory.

Now that the project file exists, from your Python code (in the same directory),
load the project so that you can read from and write to it, for example you can
store an integer: 

.. code-block:: python

    from skore import load
    project = load("my_project.skore")
    project.put("my_int", 3)

Finally, from your shell (in the same directory), start the UI locally:

.. code:: console

    python -m skore launch "my_project"

This will automatically open a browser at the UI's location:

#. On the top left, by default, you can observe that you are in a *View* called ``default``. You can rename this view or create another one.
#. From the *Items* section on the bottom left, you can add stored items to this view, either by clicking on ``+`` or by doing drag-and-drop.

For more features, please look into :ref:`auto_examples`.

.. image:: https://raw.githubusercontent.com/sylvaincom/sylvaincom.github.io/master/files/probabl/skore/2024_10_14_skore_demo.gif
   :alt: Getting started with ``skore`` demo
.. _contributing:

============
Contributing
============

First off, thanks for taking the time to contribute, everyone is welcome to
contribute.

This project is hosted on https://github.com/probabl-ai/skore.

Below are some guidelines to help you get started.

Questions, bugs and feature requests
====================================

If you have any questions, feel free to reach out:

* Join our community on `Discord <https://discord.gg/scBZerAGwW>`_ for general chat and Q&A.
* Alternatively, you can `start a discussion on GitHub <https://github.com/probabl-ai/skore/discussions>`_.

Both bugs and feature requests can be filed in the `Github issue tracker <https://github.com/probabl-ai/skore/issues>`_:

* Check if your issue does not already exist.
* For `new issues <https://github.com/probabl-ai/skore/issues/new/choose>`_, we recommend some templates but feel free to open a blank issue.
* Use `short, self-contained, correct examples <http://sscce.org/>`_.

Development
===========

Quick start
-----------

You'll need ``python >=3.9, <3.13`` to build the backend and ``Node>=20`` to build the skore-ui. Then, you can install dependencies and run the UI with:

.. code-block:: bash

    make install-skore
    make build-skore-ui
    make serve-skore-ui

You are now all setup to run the library locally.
If you want to contribute, please continue with the three other sections.

Backend
-------

Install backend dependencies with:

.. code-block:: bash

    make install-skore


You can run the API server with:

.. code-block:: bash

    make serve-skore-ui

skore-ui
--------

Install skore-ui dependencies with:

.. code-block:: bash

    npm install

in the ``skore-ui`` directory.

Run the skore-ui in dev mode (for hot-reloading) with

.. code-block:: bash

    npm run dev

in the ``skore-ui`` directory.

Then, to use the skore-ui

.. code-block:: bash

    make build-skore-ui
    make serve-skore-ui

PR format
---------

We use the `conventional commits <https://www.conventionalcommits.org/en/v1.0.0/#summary>`_ format, and we automatically check that the PR title fits this format.
In particular, commits are "sentence case", meaning "fix: Fix issue" passes, while "fix: fix issue" doesn't.

Generally the description of a commit should start with a verb in the imperative voice, so that it would properly complete the sentence: "When applied, this commit will [...]".

Examples of correct PR titles: ``docs: Update the docstrings`` or ``feat: Remove CrossValidationAggregationItem.``

Documentation
=============

Setup
-----

Our documentation uses `PyData Sphinx Theme <https://pydata-sphinx-theme.readthedocs.io/>`_.

.. warning::

    Modifications are to be done in the ``sphinx`` folder. The ``docs`` folder must *not* be touched!

To build the docs:

.. code-block:: bash

    cd sphinx
    make html

Then, you can access the local build via:

.. code-block:: bash

    open build/html/index.html

Contributing to the docstrings
------------------------------

When writing documentation, whether it be online, docstrings or help messages in the CLI and in the UI, we strive to follow some conventions that are listed below. These might be updated as time goes on.

#. The docstring will be compiled using Sphinx numpydoc so use `RST (ReStructured Text) <https://docs.open-mpi.org/en/v5.0.x/developers/rst-for-markdown-expats.html>`_ for bold, URLs, etc.
#. Argument descriptions should be written so that the following sentence makes sense: `Argument <argument> designates <argument description>`
#. Argument descriptions start with lower case, and do not end with a period or other punctuation
#. Argument descriptions start with "the" where relevant, and "whether" for booleans
#. Text is written in US english ("visualize" rather than "visualise")
#. In the CLI, positional arguments are written in snake case (``snake_case``), keyword arguments in kebab case (``kebab-case``)
#. When there is a default argument, it should be shown in the help message, typically with ``(default: <default value>)`` at the end of the message

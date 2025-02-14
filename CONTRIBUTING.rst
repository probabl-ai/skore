.. _contributing:

============
Contributing
============

Thank you for your interest in contributing to Skore! We welcome contributions from
everyone and appreciate you taking the time to get involved.

This project is hosted on https://github.com/probabl-ai/skore.

.. topic:: **Our community, our values**

    We are a community based on openness and friendly, didactic
    discussions.

    We aspire to treat everybody equally, and value their contributions.  We
    are particularly seeking people from underrepresented backgrounds in Open
    Source Software to participate and contribute their expertise and experience.

    Decisions are made based on technical merit and consensus.

    Code is not the only way to help the project. Reviewing pull
    requests, answering questions to help others on mailing lists or
    issues, organizing and teaching tutorials, working on the website,
    improving the documentation, are all priceless contributions.

    We abide by the principles of openness, respect, and consideration of
    others of the Python Software Foundation:
    https://www.python.org/psf/codeofconduct/

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
    skore create
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

Do not forget to do a hard refresh when changing the front-end (meta+shift+R) if the expected changes do not appear.

PR format
---------

We use the `conventional commits <https://www.conventionalcommits.org/en/v1.0.0/#summary>`_ format, and we automatically check that the PR title fits this format.
In particular, commits are "sentence case", meaning "fix: Fix issue" passes, while "fix: fix issue" doesn't.

Generally the description of a commit should start with a verb in the imperative voice, so that it would properly complete the sentence: "When applied, this commit will [...]".

Examples of correct PR titles: ``docs: Update the docstrings`` or ``feat: Remove CrossValidationAggregationItem.``

Tests
-----

To run the tests locally, you may run

.. code-block:: bash

    make test

To run only back-end tests, use:

.. code-block:: bash

    make test-backend

To run only front-end tests, use:

.. code-block:: bash

    make test-frontend


Linting
-------

We use the linter ruff to make sure that the code is formatted correctly.

.. code-block:: bash

    make lint


Documentation
=============

Setup
-----

Our documentation uses `PyData Sphinx Theme <https://pydata-sphinx-theme.readthedocs.io/>`_.

To build the docs:

.. code-block:: bash

    cd sphinx
    make html

Then, you can access the local build via:

.. code-block:: bash

    open build/html/index.html

The PR will also build the documentation and a bot will automatically add a comment with a link to the documentation preview to easily check the results.

Skipping examples when building the docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The examples can take a long time to build, so if you are not working on them you can instead run `make html-noplot` to avoid building them altogether.

If you are working on an example and wish to only build that one, you can do so by temporarily editing `sphinx/conf.py`. Follow `the sphinx-gallery documentation <https://sphinx-gallery.github.io/stable/configuration.html#parsing-and-executing-examples-via-matching-patterns>`_ for more information.
By default, the examples that are built are Python files that start with `plot_`.

Note that by default, if an example has not changed since the last time you built it, it will not be re-built.

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


Contributing to the examples
----------------------------

The examples are stored in the folder called `examples`. They are classified in subcategories.
They should be written in a python file, with cells marked by `# %%`. They will be automatically converted to rst files in the subfolder `sphinx/auto_examples`. This subfolder is listed in the gitignore, and cannot be pushed.

The python file should start by a docstring indicating the example name and the title.

No example should require to have large files in this repository. For example, no dataset should be stored, it should be downloaded in the script.

Contributing to the ReadMe
--------------------------

The README.md file can be modified and is part of the documentation.
This file is used to be presented on `PyPI <https://pypi.org/project/skore/#description>`_.

.. _contributing:

============
Contributing
============

Thank you for your interest in contributing to Skore! We welcome contributions from
everyone and appreciate you taking the time to get involved.

This project is hosted on https://github.com/probabl-ai/skore.

.. topic:: **Our values**

    We aspire to treat everybody equally, and value their contributions.
    We are particularly seeking people from underrepresented backgrounds in Open Source
    Software to participate and contribute their expertise and experience.

    Decisions are made based on technical merit, consensus, and roadmap priorities.

    Code is not the only way to help the project. Reviewing pull requests, answering
    questions to help others on mailing lists or issues, organizing and teaching
    tutorials, working on the website, improving the documentation, are all priceless
    contributions.

    We abide by the principles of openness, respect, and consideration of others of the
    Python Software Foundation: https://www.python.org/psf/codeofconduct/

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

You'll need ``python >=3.9, <3.13``.

Install dependencies and setup pre-commit with:

.. code-block:: bash

    make install-skore


Choosing an issue
-----------------

If you are starting to contribute to open-source, you can start by an [issue tagged `good first issue`](https://github.com/probabl-ai/skore/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22).

The implementation of some issues are not very detailed. You can either propose a solution, or choose only the issues with a Status "Ready".

Working on Issues
""""""""""""""""""

Getting Assigned to an Issue
"""""""""""""""""""""""""""

Before starting work on a pull request, please follow these steps:

1. **Find an issue to work on**: Browse the `GitHub issues <https://github.com/probabl-ai/skore/issues>`_ list. Issues labeled with `"good first issue" <https://github.com/probabl-ai/skore/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_ are particularly suitable for newcomers.

2. **Request assignment**: Comment on the issue you'd like to work on, asking to be assigned. Due to GitHub limitations, external contributors cannot assign themselves to existing issues and must request assignment from maintainers.

3. **Wait for confirmation**: A maintainer will review your request and assign the issue to you if appropriate. Assignment is typically done on a first-come, first-served basis.

4. **Begin work**: Once assigned, you can start working on the issue. Be sure to link your PR to the issue when submitting.

Issue Assignment Guidelines
"""""""""""""""""""""""""""

- **Assignment Duration**: If you've been assigned to an issue but haven't submitted a PR or provided updates for more than two weeks, maintainers may unassign you to allow others to work on it.

- **Already Assigned Issues**: If an issue is already assigned to someone, please respect that assignment unless the issue has been stalled for an extended period (usually more than two weeks without activity).

- **Issue Tags**:
  - **good first issue**: These issues are specifically selected to be accessible to newcomers.
  - **help wanted**: These issues are open for community contributions.
  - **bug**: Indicates an unexpected problem or unintended behavior.
  - **enhancement**: Suggests improvements to existing functionality.
  - **epic**: A large body of work that can be broken down into several smaller issues.
  - **user-reported**: Issues reported by users of the library.
  - **hackathon-friendly**: Issues suitable for working on during hackathons or sprints.
  - **hackathon-milestone**: Issues prioritized for completion during upcoming hackathon events.
  - **needs-triage**: Recently submitted issues that need review and categorization by maintainers.
  - **documentation**: Issues related to improving or fixing documentation.

Tips for New Contributors
"""""""""""""""""""""""""

- Start with smaller issues to familiarize yourself with the contribution workflow.
- Don't hesitate to ask for help in the issue comments if you're stuck.
- Be sure to read our code style guidelines before submitting your PR.

- **Basic Contribution Workflow**:
  
  1. Fork the repository on GitHub to your personal account
  2. Clone your fork to your local machine: ``git clone https://github.com/<your-username>/skore.git``
  3. Create a Python virtual environment: ``python -m venv venv`` and activate it
  4. Set up the development environment: ``make install-skore``
  5. Create a new branch related to the issue: ``git checkout -b new feature``
  6. Make your changes and commit them with a message related to the issue

- Run pre-commit checks before pushing your changes to catch formatting issues:

  .. code-block:: bash


- Sign your commits using the ``-S`` flag or by configuring Git to sign by default:

  .. code-block:: bash

      git commit -S -m "Your commit message"
      
      # Configure Git to always sign commits
      git config --global commit.gpgsign true

  For more details about signing commits, see our `documentation <https://docs.skore.probabl.ai/dev/contributing.html#signing-commits>`_.

- Make sure your PR title follows the `conventional commits <https://www.conventionalcommits.org/en/v1.0.0/#summary>`_ format, such as:

  ``docs: Enhance contributing guide with issue assignment information``

- Check that your PR passes all CI checks before requesting a review.
- Keep your PR focused on a single issue to simplify the review process.

Pull requests
-------------

We use the `conventional commits <https://www.conventionalcommits.org/en/v1.0.0/#summary>`_ format, and we automatically check that the PR title fits this format.

- In particular, commits are "sentence case", meaning that the ``fix: Fix issue`` title passes, while ``fix: fix issue`` does not.
- Generally, the description of a commit should start with a verb in the imperative voice, so that it would properly complete the sentence: ``When applied, this commit will [...]``.
- Examples of correct PR titles: ``docs: Update the docstrings`` or ``feat: Remove CrossValidationAggregationItem``.

Skore is a company-driven project. We might provide extensive help to bring PRs to be merged to meet internal deadlines. In such cases, we will warn you in the PR.


Tests
-----

To run the tests locally, you may run

.. code-block:: bash

    make test


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

The examples are stored in the `examples` folder:

- They are classified in subcategories.
- They should be written in a python script (`.py`), with cells marked by `# %%`, to separate code cells and markdown cells, as they will be rendered as notebooks (`.ipynb`).
- The file should start with a docstring giving the example title.
- No example should require to have large files stored in this repository. For example, no dataset should be stored, it should be downloaded in the script.
- When built (using `make html` for example), these examples will automatically be converted into rst files in the `sphinx/auto_examples` subfolder. This subfolder is listed in the gitignore and cannot be pushed.

Contributing to the README
--------------------------

The `README.md` file can be modified and is part of the documentation (although it is not included in the online documentation).
This file is used to be presented on `PyPI <https://pypi.org/project/skore/#description>`_.

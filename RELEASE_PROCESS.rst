Release process
===============

This document describes how to make a new release of the skore package; the process does not
apply to the ``skore-*-project`` packages.

Going further, we assume you have write-access to the repository and
conda-forge project page.

.. note:: We follow the semantic versioning convention:

   - Major/Minor releases are numbered X.Y.0.
   - Bugfix releases are done as needed between major/minor releases and only apply to
     the last stable version. These releases are numbered X.Y.Z.

To release a new version of skore (e.g., from 0.1.0 to 0.2.0), here are the main
steps and appropriate resources:

- Create a PR called ``release(skore): <new-version>``, which updates CHANGELOG.rst:

  - Replace "Unreleased" with the new version; make sure to add the release
    date and a link to the diff between the previous version and the new one.
  - Create a new empty "Unreleased" section which will be filled when new PRs
    come in. A template is available as a comment at the top of ``CHANGELOG.rst``.

- Test skore manually to ensure that it is in releasable state; this includes making
  sure that skore plays well with the ``skore-*-project*`` plugins.
- Merge the PR; if possible, the PR should be up-to-date with ``main``, to ensure
  that the changelog reflects the true state of the project.
- Tag the corresponding commit as ``skore/<new-version>``
- Create a `GitHub release <https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>`__ corresponding to our new tag

  - For convenience, GitHub allows you to create the tag and the release at the
    time.
  - The GitHub release description should link to the online documentation.
  - Releases must be approved by someone, usually a maintainer.

Publishing the new version on PyPI is done automatically.

Publishing the new version on conda-forge is done semi-automatically by a bot opening
a PR in https://github.com/conda-forge/skore-feedstock.
The bot is not instantaneous, so keep an eye out for the PR.
Review and merge if everything looks okay.

If the new recipe works fine, announce the release on social network channels 🎉!

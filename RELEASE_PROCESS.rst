Release process
===============

This document describes how to make a new release of skore; the process does not apply to the `skore-*-project` plugins.
Going further, we assume you have write-access to the repository and
conda-forge project page.

.. note:: We follow the scikit-learn versioning convention:

   - Major/Minor releases are numbered X.Y.0.
   - Bugfix releases are done as needed between major/minor releases and only apply to
     the last stable version. These releases are numbered X.Y.Z.

To release a new version of skore (e.g., from 0.1.0 to 0.2.0), here are the main
steps and appropriate resources:

- Create a PR called ``release: <new-version>``
	- Edit CHANGELOG.rst:
		- Replace "Unreleased" with the new version; make sure to add the release date and a link to the diff between the previous version and the new one
		- Create a new empty "Unreleased" section which will be filled when new PRs come in
- Test skore manually to ensure that it is in releasable state; this includes making sure that skore plays well with the `skore-*-project* plugins.
- Merge the PR; if possible, the PR should be up-to-date with `main`, to ensure that no changes are undocumented.
- Once the PR is merged, tag the corresponding commit as `skore/<new-version>`
- On GitHub, use the Release tool to create a release corresponding to our new tag
    - For convenience, GitHub allows you to create the tag and the release at the time.
	- The release description should link to the online documentation.

Pushing the wheel to Pypi and updating the conda-forge recipe are done automatically:
	- Updating the conda-forge recipe is done by a bot in https://github.com/conda-forge/skore-feedstock. Review and merge if everything looks okay.

If the new recipe works fine, announce the release on social network channels 🎉!

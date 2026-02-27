Release process
===============

This document is aimed at established contributors to the project.
Going further, we assume you have write-access to the repository, PyPI and
conda-forge project page.

.. note:: We follow the scikit-learn versioning convention:

   - Major/Minor releases are numbered X.Y.0.
   - Bugfix releases are done as needed between major/minor releases and only apply to
     the last stable version. These releases are numbered X.Y.Z.

To release a new minor version of skore (e.g., from 0.1.0 to 0.2.0), here are the main
steps and appropriate resources:

- Create a PR called ``release: <new-version>``
	- Edit CHANGELOG.rst:
		- Replace "Unreleased" with the new version; make sure to add the release date and a link to the diff between the previous version and the new one
		- Create a new empty "Unreleased" section which will be filled when new PRs come in
- Once the PR is merged, tag the corresponding commit as `skore/<new-version>`
- On GitHub, use the Release tool to create a release corresponding to our new tag
	- The release description should link to the online documentation.

Pushing the wheel to Pypi and updating the conda-forge recipe are done automatically:
	- Updating the conda-forge recipe is done by a bot in https://github.com/conda-forge/skore-feedstock. Review and merge if everything looks okay.

If the new recipe works fine, announce the release on social network channels 🎉!

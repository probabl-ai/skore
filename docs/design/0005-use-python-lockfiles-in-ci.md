---
date: 2025-01-31
decision-makers: ["@thomass-dev"]
---

# Use python `test-requirements.txt` lockfiles in CI

## Context and Problem Statement

With the pull-request [#916](https://github.com/probabl-ai/skore/pull/916), we cache now
the python dependencies to speed-up CI. However, to update in the cache the version of a
dependency, for example, to take advantage of a bugfix, we have to manually purge the GH
cache from the GH cli. It's not scallable.

Moreover, in state of the CI, we can't control what version of a dependency is used. It
can lead to inconsistent tests, in particular when between two runs, a new version of a
dependency is released.

We therefore want to define the dependencies versions in the CI, while leaving the user
free to install what he wants.

## Decision Outcome

We can't fix dependencies version in the `pyproject.toml` file without impacting users.
Instead, we fix the versions in separate requirement files, intended to be used only by
the CI. These files must be rebuilt after each change in the dependencies list, to stay
in sync at all times with the `pyproject.toml` file.

These files are now used to construct the python cache in the CI.
These files are automatically managed by `dependabot`, to take account of new weekly
bugfixes. It produces pull-requests that must be accepted by maintainers.

## More Information

Implementation in pull-request [#1238](https://github.com/probabl-ai/skore/pull/1238).

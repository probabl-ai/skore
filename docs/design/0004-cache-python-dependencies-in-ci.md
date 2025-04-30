---
date: 2025-01-09
decision-makers: ["@thomass-dev"]
---

# Cache python dependencies in CI

## Context and Problem Statement

The choice was made to support the 3 latests versions of `scikit-learn`, jointly with
the OS team.

To date, we already have a CI running `skore` tests with 2 OS, and 4 python versions.
The CI therefore runs the same job 2*4=8 times. To limit the installation time each time
we run the test, we use a native GH mechanism to persist the pip cache. That way, python
packages can be installed without the need to be re-download from PyPI. It's better than
nothing, but installation time is incompressibly slow.

Supporting 3 versions of `scikit-learn` means adding at least 2 additional test runs:
- the latest version is test on each OS/python versions (no additional test run),
- the 2 older versions of scikit-learn are tested on ubuntu-latest python 3.12 (2
  additional test runs)

The CI is becoming very long, we need to find a way to reduce its duration.

## Decision Outcome

Speed-up tests by installing python dependencies in virtual environments and caching it:

* pros: reduction of one minute per job when the cache is already built, otherwise
  equivalent.
* cons: cache must be purged manually to install a new version of a dependency. It's not
  a big deal for now, since the gain is higher than the constraint.

Dependencies no longer need to be installed at each step, as they are already present in
the cached venv.

We base the construction of the cache on the n-tuple OS, python, scikit-learn, and the
hash of the `pyproject.toml` file. This way, if the dependencies list changes, a new
cache is automatically built.

Each test run knows which cache to use, depending on its context.

Unused cache is automatically purged by GH after 90 days.

## More Information

Implementation in pull-request [#916](https://github.com/probabl-ai/skore/pull/916).

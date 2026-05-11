# How to bump the minimum required version of Python

- Update `requires-python` in `pyproject.toml`
- Update classifiers in `pyproject.toml`
- Remove mentions from `supported-versions.json`
- Remove `ci/requirements/skore/python-<removed_version>`
- Update `README.md`
  - Badge
  - "You need Python>=3.x"

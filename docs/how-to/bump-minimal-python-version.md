# How to bump the minimum required version of Python

Since dependencies and the CI matrix are managed by [pixi](https://pixi.sh)
(`pixi.toml` / `pixi.lock`), bumping the minimum supported Python version touches both
the package metadata and the pixi configuration.

- Update `skore/pyproject.toml`
  - `requires-python`
  - the `Programming Language :: Python :: 3.x` classifiers
- Update `pixi.toml`
  - remove the `[feature.py3xx]` table for the dropped version
  - update the `[environments]` table to drop every environment relying on the removed
    `py3xx` feature
- Update `.github/workflows/pytest.yml`
  - update the `matrix.environment` list (and any `exclude` / `include` entries) so it
    stays in sync with the `[environments]` table of `pixi.toml`
- Refresh the lockfile so it matches the new configuration:

  ```bash
  pixi install
  # or, to also pick up newer dependency versions:
  pixi update
  ```

- Update `README.md`
  - the Python badge
  - the "You need `python>=3.x`" mention
  - the "Support" section listing the tested Python/scikit-learn versions
- Update `CONTRIBUTING.rst`
  - the "You will need `python >=3.x`" mention

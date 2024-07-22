# mandr

![lint and test](https://github.com/probabl-ai/mandr/actions/workflows/lint-and-test.yml/badge.svg)

A service to send data into.

## Example

```python
from mandr import InfoMander

# We use paths to explain where the dictionary is stored
mander = InfoMander('/org/usecase/train/1')
mander.add_info(...)
mander.add_logs(...)
mander.add_templates(...)
mander.add_views(...)
```

## Development

Install dependencies with
```sh
make install
```

You can run the API server with
```sh
make serve-api
```

When dependencies are changed in `pyproject.toml` the lockfiles should be updated via [`pip-compile`](https://github.com/jazzband/pip-tools):
```sh
make pip-compile
```

## Documentation

For now, the documentation must be build locally to be accessed.

```sh
make build-doc
```

Then, you can access the local build via:
```sh
open doc/_build/html/index.html
```

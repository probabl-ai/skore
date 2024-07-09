# mandr

![lint and test](https://github.com/probabl-ai/mandr/actions/workflows/lint-and-test.yml/badge.svg)

Service to send data into, install via

```
python -m pip install -e
python -m pip install flask diskcache pandas altair
```


```python
from mandr import InfoMander

# We use paths to explain where the dictionary is stored
mander = InfoMander('/org/usecase/train/1')
mander.add_info(...)
mander.add_logs(...)
mander.add_templates(...)
mander.add_views(...)
```

When ready to view, run flask:

```
python -m flask --app mandr.app run --reload
```

## Dependencies

When dependencies are changed in `pyproject.toml` the lockfiles should be updated by running [`pip-compile`](https://github.com/jazzband/pip-tools):
```sh
pip-compile --output-file=requirements.txt pyproject.toml
pip-compile --extra=test --output-file=requirements-test.txt pyproject.toml
```

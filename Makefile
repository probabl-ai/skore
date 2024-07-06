pip-compile:
	pip-compile --output-file=requirements.txt pyproject.toml
	pip-compile --extra=test --output-file=requirements-test.txt pyproject.toml

install:
	python -m pip install -r requirements-test.txt
	python -m pip install -e .
	pre-commit install

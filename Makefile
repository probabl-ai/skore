pip-compile:
	pip-compile --output-file=requirements.txt pyproject.toml
	pip-compile --extra=test --output-file=requirements-test.txt pyproject.toml

install:
	python -m pip install -e . -r requirements.txt -r requirements-test.txt
	pre-commit install

check-wip:
	pre-commit run --all-files
    python -m pytest tests

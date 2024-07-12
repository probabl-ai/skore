pip-compile:
	pip-compile --output-file=requirements.txt pyproject.toml
	pip-compile --extra=test --output-file=requirements-test.txt pyproject.toml

install:
	python -m pip install -e . -r requirements.txt -r requirements-test.txt
	pre-commit install

serve-dashboard-api:
	python -m uvicorn mandr.dashboard.webapp:app --reload --reload-dir src --host 0.0.0.0 --timeout-graceful-shutdown 0

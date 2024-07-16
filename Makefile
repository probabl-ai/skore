pip-compile:
	pip-compile --output-file=requirements.txt pyproject.toml
	pip-compile --extra=test --output-file=requirements-test.txt pyproject.toml

install:
	python -m pip install -e . -r requirements.txt -r requirements-test.txt
	pre-commit install

check-wip:
	pre-commit run --all-files
	python -m pytest tests

serve-api:
	python -m uvicorn mandr.dashboard.webapp:app --reload --reload-dir ./src --host 0.0.0.0 --timeout-graceful-shutdown 0

build-frontend:
	# build the SPA
	cd frontend && npm install
	cd frontend && npm run build
	# empty app static folder except gitignore
	find src/mandr/dashboard/static -mindepth 1 -maxdepth 1 ! -name ".gitignore" -exec rm -r -- {} +
	cp -a frontend/dist/. src/mandr/dashboard/static
	rm -rf frontend/dist

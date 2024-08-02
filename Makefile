pip-compile:
	pip-compile --output-file=requirements.txt pyproject.toml
	pip-compile --extra=test --output-file=requirements-test.txt pyproject.toml
	pip-compile --extra=tools --output-file=requirements-tools.txt pyproject.toml
	pip-compile --extra=doc --output-file=requirements-doc.txt pyproject.toml

install:
	python -m pip install -e . -r requirements.txt -r requirements-test.txt -r requirements-tools.txt
	pre-commit install

check-wip:
	pre-commit run --all-files
	python -m pytest tests

serve-api:
	MANDR_ROOT=.datamander python -m uvicorn mandr.dashboard.webapp:app --reload --reload-dir ./src --host 0.0.0.0 --timeout-graceful-shutdown 0

build-frontend:
	# build the SPA
	cd frontend && npm install
	cd frontend && npm run build
	# empty app static folder
	rm -rf src/mandr/dashboard/static
	cp -a frontend/dist/. src/mandr/dashboard/static
	rm -rf frontend/dist

build-doc:
	python -m pip install -e . -r requirements-doc.txt
	cd doc
	# Run a make instruction inside the doc folder
	$(MAKE) -C doc html

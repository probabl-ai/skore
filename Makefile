MANDR_ROOT ?= ".datamander"

pip-compile:
	pip-compile --output-file=requirements.txt pyproject.toml
	pip-compile --extra=test --output-file=requirements-test.txt pyproject.toml
	pip-compile --extra=tools --output-file=requirements-tools.txt pyproject.toml
	pip-compile --extra=doc --output-file=requirements-doc.txt pyproject.toml

install:
	python -m pip install -e . -r requirements.txt -r requirements-test.txt -r \
	requirements-tools.txt -r requirements-doc.txt
	pre-commit install

check-wip:
	pre-commit run --all-files
	python -m pytest tests

serve-api:
	MANDR_ROOT=$(MANDR_ROOT) python -m uvicorn \
		--factory mandr.api:create_api_app \
		--reload --reload-dir ./src \
		--host 0.0.0.0 \
		--port 22140 \
		--timeout-graceful-shutdown 0

serve-dashboard:
	MANDR_ROOT=$(MANDR_ROOT) python -m uvicorn \
		--factory mandr.dashboard:create_dashboard_app \
		--reload --reload-dir ./src \
		--host 0.0.0.0 \
		--port 22140 \
		--timeout-graceful-shutdown 0

build-frontend:
	# build the SPA
	cd frontend && npm install
	cd frontend && npm run build
	# empty app static folder
	rm -rf src/mandr/dashboard/static
	cp -a frontend/dist/. src/mandr/dashboard/static
	# build the sharing library
	cd frontend && npm run build:lib
	cp -a frontend/dist/. src/mandr/dashboard/static
	# clean up
	rm -rf frontend/dist

build-doc:
	python -m pip install -e . -r requirements-doc.txt
	cd doc
	# Run a make instruction inside the doc folder
	$(MAKE) -C doc html

build-package: build-frontend
	python -m pip install build
	python -m build

release-package: build-package
	python -m pip install twine
	python -m twine upload --repository testpypi dist/*

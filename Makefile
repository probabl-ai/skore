SKORE_ROOT ?= ".datamander"

pip-compile:
	pip-compile --output-file=requirements.txt pyproject.toml
	pip-compile --extra=test --output-file=requirements-test.txt pyproject.toml
	pip-compile --extra=tools --output-file=requirements-tools.txt pyproject.toml

install:
	python -m pip install \
		-e . \
		-r requirements.txt \
		-r requirements-test.txt \
		-r requirements-tools.txt
	pre-commit install

check-wip:
	pre-commit run --all-files
	python -m pytest tests

serve-api:
	SKORE_ROOT=$(SKORE_ROOT) python -m uvicorn \
		--factory skore.ui.app:create_app \
		--reload --reload-dir ./src \
		--host 0.0.0.0 \
		--port 22140 \
		--timeout-graceful-shutdown 0

serve-ui:
	SKORE_ROOT=$(SKORE_ROOT) python -m uvicorn \
		--factory skore.ui.app:create_app \
		--reload --reload-dir ./src \
		--host 0.0.0.0 \
		--port 22140 \
		--timeout-graceful-shutdown 0

build-frontend:
	# build the SPA
	cd frontend && npm install
	cd frontend && npm run build
	# empty app static folder
	rm -rf src/skore/ui/static
	cp -a frontend/dist/. src/skore/ui/static
	# build the sharing library
	cd frontend && npm run build:lib
	cp -a frontend/dist/. src/skore/ui/static
	# clean up
	rm -rf frontend/dist

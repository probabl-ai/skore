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
	# cleanup
	rm -rf frontend/dist
	rm -rf src/skore/ui/static
	# build
	(\
		cd frontend;\
		npm install;\
		npm run build;\
		npm run build:lib -- --emptyOutDir false;\
	)
	# move
	mv frontend/dist/ src/skore/ui/static

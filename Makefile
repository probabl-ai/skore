SKORE_ROOT ?= ".datamander"

pip-compile:
	python -m piptools compile --output-file=skore/requirements.txt skore/pyproject.toml
	python -m piptools compile --extra=test --output-file=skore/requirements-test.txt skore/pyproject.toml
	python -m piptools compile --extra=tools --output-file=skore/requirements-tools.txt skore/pyproject.toml

install-skore:
	python -m pip install \
		-e skore/ \
		-r skore/requirements.txt \
		-r skore/requirements-test.txt \
		-r skore/requirements-tools.txt

	pre-commit install

build-skore-ui:
	# cleanup
	rm -rf skore-ui/dist
	rm -rf skore/src/skore/ui/static
	# build
	( \
		cd skore-ui; \
		npm install; \
		npm run build; \
		npm run build:lib -- --emptyOutDir false; \
	)
	# move
	mv skore-ui/dist/ skore/src/skore/ui/static

serve-skore-ui:
	SKORE_ROOT=$(SKORE_ROOT) python -m uvicorn \
		--factory skore.ui.app:create_app \
		--reload --reload-dir skore/src \
		--host 0.0.0.0 \
		--port 22140 \
		--timeout-graceful-shutdown 0

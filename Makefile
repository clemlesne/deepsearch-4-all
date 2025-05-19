.PHONY: version version-full install upgrade test dev build lint

# Versioning
version_full ?= $(shell $(MAKE) --silent version-full)
version_small ?= $(shell $(MAKE) --silent version)
# App location
default_location := swedencentral
openai_location := swedencentral
search_location := westeurope
# Container configuration
container_name := ghcr.io/clemlesne/deepsearch-4-all
# Bicep outputs
api_url ?= $(shell az deployment sub show --name $(name) | yq '.properties.outputs["apiUrl"].value')
blob_account_name ?= $(shell az deployment sub show --name $(name) | yq '.properties.outputs["blobAccountName"].value')
front_url ?= $(shell az deployment sub show --name $(name) | yq '.properties.outputs["frontUrl"].value')
static_url ?= $(shell az deployment sub show --name $(name) | yq '.properties.outputs["staticUrl"].value')

version:
	@bash ./cicd/version/version.sh -g . -c

version-full:
	@bash ./cicd/version/version.sh -g . -c -m

install:
	@echo "➡️ Installing venv..."
	uv venv --python 3.13 --allow-existing

	@echo "➡️ Installing crawl4ai dependencies..."
	uv run playwright install

	$(MAKE) install-deps

install-deps:
	@echo "➡️ Syncing dependencies..."
	uv sync --extra dev

upgrade:
	@echo "➡️ Updating Git submodules..."
	git submodule update --init --recursive

	@echo "➡️ Compiling requirements..."
	uv lock --upgrade

test:
	$(MAKE) test-static
	$(MAKE) test-unit

test-static:
	@echo "➡️ Test dependencies issues (deptry)..."
	uv run deptry app

	@echo "➡️ Test code smells (Ruff)..."
	uv run ruff check

	@echo "➡️ Test types (Pyright)..."
	uv run pyright

test-unit:
	@echo "➡️ Unit tests (Pytest)..."
	uv run pytest \
		--junit-xml=test-reports/$(version_full)-simple.xml \
		--log-file=test-reports/$(version_full)-simple.log \
		--maxprocesses=2 \
		-n=logical \
		tests/*.py

dev:
	VERSION=$(version_full) CI=true uv run gunicorn app.main:api \
		--access-logfile - \
		--bind 0.0.0.0:8080 \
		--graceful-timeout 60 \
		--proxy-protocol \
		--reload \
		--reload-extra-file .env \
		--timeout 60 \
		--worker-class uvicorn.workers.UvicornWorker \
		--workers 2

lint:
	@echo "➡️ Fix with formatter..."
	uv run ruff format

	@echo "➡️ Lint with linter..."
	uv run ruff check --fix

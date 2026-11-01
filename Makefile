.DEFAULT_GOAL := help

QUARTO ?= uvx --from quarto-cli==1.10.18 quarto
HOST ?= 127.0.0.1
PORT ?= 4200

.PHONY: help prepare dev preview render check check-docs clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

prepare: ## Generate static evidence pages from local source and recorded results.
	uv run --project docs python -m evidence.render

dev: ## Preview the static site with live reload.
	$(QUARTO) preview docs --no-browser --host $(HOST) --port $(PORT)

preview: dev ## Alias for dev.

render: ## Render static HTML and Typst diagrams into docs/site/.
	$(QUARTO) render docs --no-execute

check: prepare ## Check static authoring, API references, and evidence reporting.
	uv run --project docs ruff format --check --config docs/pyproject.toml docs/check.py docs/tests evidence
	uv run --project docs ruff check --config docs/pyproject.toml docs/check.py docs/tests evidence
	uv run --project docs pytest -c docs/pyproject.toml docs/tests evidence/tests
	uv run --project docs python -m docs.check

check-docs: render ## Validate the complete rendered site.
	uv run --project docs python -m docs.check --rendered

clean: ## Remove generated pages, site output, diagrams, and tool caches.
	uv run --project docs python -m evidence.render --clean
	rm -rf docs/site docs/.quarto docs/assets/diagrams docs/.pytest_cache docs/.ruff_cache
	rm -f docs/.gitignore
	rm -rf docs/__pycache__ docs/tests/__pycache__ evidence/__pycache__
	rm -rf evidence/tests/__pycache__ evidence/tests/runtime/__pycache__
	rm -rf evidence/experiments/__pycache__ evidence/experiments/*/__pycache__

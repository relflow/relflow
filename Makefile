.DEFAULT_GOAL := help

HOST ?= 127.0.0.1
PORT ?= 4300
DOCS := docs/starlight

.PHONY: help prepare dev preview render check check-python check-docs check-browser clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

prepare: ## Install the locked static documentation tools.
	uv sync --project docs --locked
	npm --prefix $(DOCS) ci

dev: ## Preview the static site with live reload.
	npm --prefix $(DOCS) run dev -- --host $(HOST) --port $(PORT)

preview: ## Serve the built site locally.
	npm --prefix $(DOCS) run preview -- --host $(HOST) --port $(PORT)

render: ## Build static HTML and search into docs/starlight/dist/.
	npm --prefix $(DOCS) run build

check-python: ## Check the static publisher and Evidence tooling.
	uv run --project docs ruff format --check --config docs/pyproject.toml docs/publish.py docs/tests evidence
	uv run --project docs ruff check --config docs/pyproject.toml docs/publish.py docs/tests evidence
	uv run --project docs pytest -c docs/pyproject.toml docs/tests evidence/tests

check: check-python ## Check documentation formatting, types, and static contracts.
	npm --prefix $(DOCS) run format:check
	npm --prefix $(DOCS) run check
	npm --prefix $(DOCS) test

check-docs: render ## Check built links, canonical URLs, and page/bundle budgets.
	uv run --project docs python -m docs.publish --check
	npm --prefix $(DOCS) run check:performance

check-browser: check-docs ## Check desktop and mobile browser interactions.
	npm --prefix $(DOCS) run test:browser

clean: ## Remove generated site output and tool caches.
	rm -rf docs/starlight/dist docs/starlight/.astro docs/starlight/performance-report
	rm -rf docs/starlight/playwright-report docs/starlight/test-results
	rm -rf docs/.pytest_cache docs/.ruff_cache
	rm -rf docs/__pycache__ docs/tests/__pycache__ evidence/__pycache__
	rm -rf evidence/tests/__pycache__ evidence/tests/runtime/__pycache__
	rm -rf evidence/experiments/__pycache__ evidence/experiments/*/__pycache__

.DEFAULT_GOAL := help

QUARTO ?= uvx --from quarto-cli==1.9.38 quarto
HOST ?= 127.0.0.1
PORT ?= 4200
PREVIEW_FLAGS ?= --no-browser --host $(HOST) --port $(PORT)

.PHONY: help dev preview render check-docs proofs build clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Targets:"} /^[a-zA-Z_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dev: ## Start the Quarto preview server with live reload.
	$(QUARTO) preview docs $(PREVIEW_FLAGS)

preview: dev ## Alias for dev.

render: ## Render the static docs site into docs/site/.
	$(QUARTO) render docs

check-docs: ## Validate docs quietly without touching the workspace.
	@set -eu; \
		DOCS_CHECK_DIR="$$(mktemp -d)"; \
		trap 'rm -rf "$${DOCS_CHECK_DIR:?}"' EXIT HUP INT TERM; \
		rsync -a --exclude='site/' --exclude='.quarto/' --exclude='__pycache__/' --exclude='assets/diagrams/' docs/ "$$DOCS_CHECK_DIR/docs/"; \
		$(QUARTO) render "$$DOCS_CHECK_DIR/docs" --quiet; \
		if grep -r -l --include='*.html' 'class="typst-render-error"' "$$DOCS_CHECK_DIR/docs/site"; then \
			printf '%s\n' 'Typst diagram rendering failed.' >&2; \
			exit 1; \
		fi

proofs: ## Run synthetic proofs of learned model behavior.
	uv run pytest -n 0 proofs

build: render ## Alias for render.

clean: ## Remove rendered docs output.
	rm -rf docs/site

.DEFAULT_GOAL := help
.PHONY: help build build-release check fmt fmt-check clippy test check-all clean doc doc-private serve-docs
CARGO ?= cargo
PYTHON ?= python3
DOC_PORT ?= 8000
DOC_HOST ?= 127.0.0.1
DOC_CRATE ?= omeco

help:
	@printf "Targets:\n"
	@printf "  build         Build the project\n"
	@printf "  build-release Build release binaries\n"
	@printf "  check         Run cargo check\n"
	@printf "  fmt           Format code\n"
	@printf "  fmt-check     Check formatting\n"
	@printf "  clippy        Run clippy (deny warnings)\n"
	@printf "  test          Run the test suite\n"
	@printf "  check-all     Run fmt-check, clippy, and test\n"
	@printf "  doc           Build rustdoc and open it\n"
	@printf "  doc-private   Build rustdoc including private items\n"
	@printf "  serve-docs    Serve rustdoc at http://%s:%s/%s\n" "$(DOC_HOST)" "$(DOC_PORT)" "$(DOC_CRATE)"
	@printf "  clean         Clean build artifacts\n"

build:
	$(CARGO) build

build-release:
	$(CARGO) build --release

check:
	$(CARGO) check

fmt:
	$(CARGO) fmt

fmt-check:
	$(CARGO) fmt -- --check

clippy:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

test:
	$(CARGO) test --all-features

check-all: fmt-check clippy test
	@echo "All checks passed."

doc:
	$(CARGO) doc --no-deps --all-features --open

doc-private:
	$(CARGO) doc --no-deps --document-private-items --open

serve-docs:
	$(CARGO) doc --no-deps --all-features
	@echo "Serving rustdoc at http://$(DOC_HOST):$(DOC_PORT)/$(DOC_CRATE)"
	$(PYTHON) -m http.server $(DOC_PORT) --directory target/doc --bind $(DOC_HOST)

clean:
	$(CARGO) clean

.DEFAULT_GOAL := help
.PHONY: help build build-release check fmt fmt-check clippy test check-all clean doc doc-private serve-docs python-dev python-build python-test

CARGO ?= cargo
PYTHON ?= python3
DOC_PORT ?= 8000
DOC_HOST ?= 127.0.0.1
DOC_CRATE ?= omeco

help:
	@printf "Rust targets:\n"
	@printf "  build         Build the workspace\n"
	@printf "  build-release Build release binaries\n"
	@printf "  check         Run cargo check\n"
	@printf "  fmt           Format code\n"
	@printf "  fmt-check     Check formatting\n"
	@printf "  clippy        Run clippy (deny warnings)\n"
	@printf "  test          Run the test suite\n"
	@printf "  check-all     Run fmt-check, clippy, and test\n"
	@printf "  doc           Build rustdoc and open it\n"
	@printf "  serve-docs    Serve rustdoc at http://%s:%s/%s\n" "$(DOC_HOST)" "$(DOC_PORT)" "$(DOC_CRATE)"
	@printf "  clean         Clean build artifacts\n"
	@printf "\nPython targets:\n"
	@printf "  python-dev    Build and install Python package locally\n"
	@printf "  python-build  Build Python wheel\n"
	@printf "  python-test   Run Python tests\n"

build:
	$(CARGO) build --workspace

build-release:
	$(CARGO) build --workspace --release

check:
	$(CARGO) check --workspace

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings

test:
	$(CARGO) test --workspace --all-features

check-all: fmt-check clippy test
	@echo "All checks passed."

doc:
	$(CARGO) doc --no-deps --all-features --open -p omeco

doc-private:
	$(CARGO) doc --no-deps --document-private-items --open -p omeco

serve-docs:
	$(CARGO) doc --no-deps --all-features -p omeco
	@echo "Serving rustdoc at http://$(DOC_HOST):$(DOC_PORT)/$(DOC_CRATE)"
	$(PYTHON) -m http.server $(DOC_PORT) --directory target/doc --bind $(DOC_HOST)

clean:
	$(CARGO) clean

# Python targets
python-dev:
	cd omeco-python && maturin develop

python-build:
	cd omeco-python && maturin build --release

python-test: python-dev
	cd omeco-python && $(PYTHON) -m pytest -v

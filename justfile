# Prism task runner — lint / format / test
# Usage: just          (lists recipes)
#        just check    (fmt-check + lint + test)
#        just fix      (format + lint --fix)

set dotenv-load := false
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Prefer project venv; fall back to PATH.
python := if os() == "windows" {
  if path_exists(".venv/Scripts/python.exe") == "true" {
    ".venv/Scripts/python.exe"
  } else {
    "python"
  }
} else {
  if path_exists(".venv/bin/python") == "true" {
    "./.venv/bin/python"
  } else {
    "python"
  }
}

src_dirs := "src tests"

default:
    @just --list

# --- quality ---

# Ruff linter (no writes)
lint:
    {{ python }} -m ruff check {{ src_dirs }}

# Auto-fix lint issues where safe
lint-fix:
    {{ python }} -m ruff check --fix {{ src_dirs }}

# Format source (writes)
fmt:
    {{ python }} -m ruff format {{ src_dirs }}

# Fail if formatting needed (CI)
fmt-check:
    {{ python }} -m ruff format --check {{ src_dirs }}

# Format + safe lint fixes
fix: fmt lint-fix

# Unit tests
test *args:
    {{ python }} -m pytest -q {{ args }}

# Full gate: format check + lint + tests
check: fmt-check lint test

# Alias used by some CI templates
ci: check

# --- project helpers (parity with Makefile) ---

install:
    {{ python }} -m pip install -e ".[dev]"

init-db:
    {{ python }} -m prism.cli init-db

rebuild-bm25:
    {{ python }} -m prism.cli rebuild-bm25

ingest-demo:
    {{ python }} -m prism.cli ingest-demo

golden target="240":
    {{ python }} -m prism.cli golden build --target {{ target }}

eval-smoke:
    {{ python }} -m prism.cli eval run --config configs/eval/smoke.yaml --skip-generation

eval-hybrid:
    {{ python }} -m prism.cli eval run --config configs/eval/ablate_hybrid.yaml --skip-generation

up:
    docker compose up -d --build

down:
    docker compose down

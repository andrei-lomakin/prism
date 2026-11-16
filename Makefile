.PHONY: install up down logs demo test lint format migrate golden ingest-demo

install:
	pip install -e ".[dev]"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api worker

migrate:
	prism init-db

golden:
	prism golden build --target 240

ingest-demo:
	prism ingest-demo

eval-smoke:
	prism eval run --config configs/eval/smoke.yaml --skip-generation

eval-baseline:
	prism eval run --config configs/eval/baseline.yaml

demo: up golden
	@echo "Open http://127.0.0.1:8000 — then: make ingest-demo (needs Postgres)"

# Prefer `just check` / `just fix` (see justfile). Makefile kept for parity.
test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

check:
	ruff format --check src tests
	ruff check src tests
	pytest -q

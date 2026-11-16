.PHONY: install up down logs demo test lint format migrate

install:
	pip install -e ".[dev]"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api worker

migrate:
	python -c "from prism.db.session import init_db; init_db()"

demo: up
	@echo "Open http://localhost:8000 — upload a document, then ask a question on /search"

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

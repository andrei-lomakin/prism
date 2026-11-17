"""arq worker settings."""

from __future__ import annotations

from arq.connections import RedisSettings

from prism.config import get_settings
from prism.db.session import init_db
from prism.worker.tasks import ingest_document, run_eval_job


async def on_startup(ctx: dict) -> None:
    init_db()


class WorkerSettings:
    functions = [ingest_document, run_eval_job]
    on_startup = on_startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 2
    # Eval can run for many minutes; keep job visible in Redis.
    job_timeout = 3600

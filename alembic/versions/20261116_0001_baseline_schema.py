"""Baseline schema note — tables created via init_db/create_all in Stage 0–7.

Revision ID: 20261116_0001
Revises:
Create Date: 2026-11-16
"""

from __future__ import annotations

from typing import Sequence, Union

revision: str = "20261116_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing deployments use Base.metadata.create_all (prism init-db).
    # Future schema changes should be real Alembic ops from here.
    pass


def downgrade() -> None:
    pass

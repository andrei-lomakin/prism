"""Optional single-user HTTP Basic auth (TZ §8). Disabled when AUTH_PASSWORD empty."""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from prism.config import get_settings

security = HTTPBasic(auto_error=False)


def require_user(
    credentials: HTTPBasicCredentials | None = Depends(security),
) -> str | None:
    settings = get_settings()
    if not settings.auth_password:
        return None  # auth disabled
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    user_ok = secrets.compare_digest(credentials.username, settings.auth_username)
    pass_ok = secrets.compare_digest(credentials.password, settings.auth_password)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

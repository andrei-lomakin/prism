from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from prism.api.auth import require_user
from prism.config import Settings


def test_auth_disabled_when_password_empty(monkeypatch):
    monkeypatch.setattr(
        "prism.api.auth.get_settings",
        lambda: Settings(auth_password=""),
    )
    assert require_user(None) is None


def test_auth_rejects_missing_credentials(monkeypatch):
    monkeypatch.setattr(
        "prism.api.auth.get_settings",
        lambda: Settings(auth_password="secret", auth_username="prism"),
    )
    try:
        require_user(None)
        raise AssertionError("expected 401")
    except HTTPException as exc:
        assert exc.status_code == 401


def test_auth_accepts_valid(monkeypatch):
    monkeypatch.setattr(
        "prism.api.auth.get_settings",
        lambda: Settings(auth_password="secret", auth_username="prism"),
    )
    creds = HTTPBasicCredentials(username="prism", password="secret")
    assert require_user(creds) == "prism"

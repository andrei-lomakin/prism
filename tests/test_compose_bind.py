from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_compose_publishes_loopback_and_forwards_auth():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:8000:8000" in text
    assert "127.0.0.1:5434:5432" in text
    assert "AUTH_PASSWORD: ${AUTH_PASSWORD:-}" in text
    assert '"8000:8000"' not in text
    assert '"5434:5432"' not in text

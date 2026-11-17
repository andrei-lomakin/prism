import socket

import httpx
import pytest

from prism.ingest.parsers import UnsafeUrlError, assert_public_http_url, download_url_to_file
from prism.net.safe_http import read_capped, safe_get


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://10.1.2.3/x",
        "http://192.168.0.5/x",
        "http://172.16.0.5/x",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost/x",
        "http://[::1]/x",
        "http://[::ffff:127.0.0.1]/x",
        "http://[::ffff:10.1.2.3]/x",
        "http://[::ffff:169.254.169.254]/x",
        "http://[fd00::1]/x",
        "http://[fe80::1]/x",
        "http://100.64.0.1/x",
        "http://2130706433/x",
        "http://0.0.0.0/x",
        "file:///etc/passwd",
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
)
def test_blocks_non_public_targets(url):
    with pytest.raises(UnsafeUrlError):
        assert_public_http_url(url)


def test_allows_public_literal():
    assert assert_public_http_url("https://93.184.216.34/doc") == "https://93.184.216.34/doc"
    mapped = "http://[::ffff:93.184.216.34]/doc"
    assert assert_public_http_url(mapped) == mapped


def test_redirect_to_loopback_is_not_fetched(monkeypatch, tmp_path):
    def fake_getaddrinfo(host, *_args, **_kwargs):
        if host == "example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        raise socket.gaierror(host)

    monkeypatch.setattr("prism.net.safe_http.socket.getaddrinfo", fake_getaddrinfo)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        with pytest.raises(UnsafeUrlError):
            download_url_to_file("http://example.com/start", tmp_path / "out", client=client)
    assert calls == ["example.com"]


def test_download_public_body(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="hello document")

    dest = tmp_path / "out"
    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        saved = download_url_to_file("http://93.184.216.34/a.txt", dest, client=client)
    assert saved.read_text(encoding="utf-8") == "hello document"


def test_redirect_loop_stops(monkeypatch):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": str(request.url)})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        with pytest.raises(UnsafeUrlError, match="redirect loop"):
            safe_get("http://93.184.216.34/loop", client=client)
    assert len(calls) == 1


def test_too_many_redirects():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        n = len(calls)
        return httpx.Response(302, headers={"location": f"http://93.184.216.34/r{n}"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        with pytest.raises(UnsafeUrlError, match="too many redirects"):
            safe_get("http://93.184.216.34/start", client=client)
    assert len(calls) == 6


def test_read_capped_stops_during_read():
    yielded: list[int] = []

    def chunks():
        for i in range(10):
            yielded.append(i)
            yield b"x" * 1000

    with pytest.raises(UnsafeUrlError, match="exceeds"):
        read_capped(chunks(), 2500)
    assert yielded == [0, 1, 2]


def test_safe_get_rejects_non_200_without_reading_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, content=b"challenge")

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        with pytest.raises(httpx.HTTPStatusError, match="HTTP 202"):
            safe_get("http://93.184.216.34/wait", client=client)


def test_safe_get_rejects_oversized_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 1000)

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        with pytest.raises(UnsafeUrlError, match="exceeds"):
            safe_get("http://93.184.216.34/big", max_bytes=100, client=client)


def test_dns_pin_ignores_rebind(monkeypatch):
    calls = {"n": 0}
    seen: list[str] = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        calls["n"] += 1
        ip = "93.184.216.34" if calls["n"] == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    monkeypatch.setattr("prism.net.safe_http.socket.getaddrinfo", fake_getaddrinfo)

    def handler(request: httpx.Request) -> httpx.Response:
        infos = socket.getaddrinfo(request.url.host, request.url.port or 80)
        seen.append(infos[0][4][0])
        return httpx.Response(200, content=b"ok")

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
        body, _ctype, _final = safe_get("http://example.com/x", client=client)
    assert body == b"ok"
    assert seen == ["93.184.216.34"]
    assert calls["n"] == 1

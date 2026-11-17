import socket

import httpx
import pytest

from prism.ingest.parsers import UnsafeUrlError, assert_public_http_url, download_url_to_file


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
        "http://[fd00::1]/x",
        "http://[fe80::1]/x",
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


def test_redirect_to_loopback_is_not_fetched(monkeypatch, tmp_path):
    def fake_getaddrinfo(host, *_args, **_kwargs):
        if host == "example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        raise socket.gaierror(host)

    monkeypatch.setattr("prism.ingest.parsers.socket.getaddrinfo", fake_getaddrinfo)
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

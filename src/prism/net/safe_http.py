"""One public-only HTTP fetcher for ingest downloads and agent page reads.

DNS is resolved once per hop. ``socket.getaddrinfo`` is pinned to that address
for the duration of the connection so a later lookup cannot swap in a
private target (DNS rebinding). The pin is process-local and covers the
request itself, not a separate egress firewall.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urljoin, urlparse

import httpx

_MAX_REDIRECTS = 5
_MAX_URL_BYTES = 20_000_000
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_BLOCKED_HOSTS = frozenset(
    {"localhost", "localhost.localdomain", "metadata.google.internal", "metadata.internal"}
)


class UnsafeUrlError(ValueError):
    """URL target is not a public http(s) host (SSRF guard)."""


def _parse_host_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if host.isdigit():
        try:
            return ipaddress.ip_address(int(host))
        except ValueError:
            return None
    return None


def _blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return _blocked_ip(mapped)
    if ip.version == 4 and (
        ip in ipaddress.ip_network("0.0.0.0/8") or ip in ipaddress.ip_network("100.64.0.0/10")
    ):
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _assert_host_public(host: str, *, resolve: bool = True) -> str:
    """Return the IP a later connect must use. Empty string when resolve is off."""
    name = host.lower().rstrip(".")
    if (
        not name
        or name in _BLOCKED_HOSTS
        or name.endswith(".localhost")
        or name.endswith(".local")
        or name.endswith(".internal")
    ):
        raise UnsafeUrlError(f"blocked host {host}")
    literal = _parse_host_ip(name)
    if literal is not None:
        if _blocked_ip(literal):
            raise UnsafeUrlError(f"blocked address {literal}")
        return str(literal)
    if not resolve:
        return ""
    try:
        infos = socket.getaddrinfo(name, None)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"DNS failed for {host}: {exc}") from exc
    addresses = [item[4][0].split("%", 1)[0] for item in infos if item[4]]
    if not addresses:
        raise UnsafeUrlError(f"no addresses for {host}")
    for addr in addresses:
        ip = ipaddress.ip_address(addr)
        if _blocked_ip(ip):
            raise UnsafeUrlError(f"blocked address {ip} for host {host}")
    return addresses[0]


def assert_public_http_url(url: str, *, resolve: bool = True) -> str:
    """Reject non-http(s) URLs and loopback / private / link-local targets."""
    _split_public_url(url, resolve=resolve)
    return url


def _split_public_url(url: str, *, resolve: bool) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError(f"unsupported scheme {parsed.scheme or '(none)'}")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("credentials in URL are not allowed")
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError("missing host")
    ip = _assert_host_public(host, resolve=resolve)
    return host, ip


def public_connect_target(url: str) -> tuple[str, str]:
    """Validate ``url`` with a single DNS lookup and return ``(host, ip)``."""
    host, ip = _split_public_url(url, resolve=True)
    if not ip:
        raise UnsafeUrlError(f"no addresses for {host}")
    return host, ip


def _addrinfo_for(ip: str, port: int | None) -> list[tuple]:
    port_n = int(port or 0)
    addr = ipaddress.ip_address(ip)
    if addr.version == 6:
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port_n, 0, 0))]
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port_n))]


@contextmanager
def _pin_getaddrinfo(host: str, ip: str) -> Iterator[None]:
    """Force lookups of ``host`` to ``ip`` until the connection attempt returns."""
    host_key = host.lower().rstrip(".")
    original = socket.getaddrinfo

    def pinned(h, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        raw = h.decode() if isinstance(h, bytes) else str(h)
        if raw.lower().rstrip(".") == host_key:
            return _addrinfo_for(ip, port)
        return original(h, port, *args, **kwargs)

    socket.getaddrinfo = pinned
    try:
        yield
    finally:
        socket.getaddrinfo = original


def read_capped(chunks: Iterator[bytes], max_bytes: int) -> bytes:
    """Read until ``max_bytes``. Stop pulling chunks once the cap is crossed."""
    buf = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        if len(buf) + len(chunk) > max_bytes:
            raise UnsafeUrlError(f"response exceeds {max_bytes} bytes")
        buf.extend(chunk)
    return bytes(buf)


def safe_get(
    url: str,
    *,
    timeout: float = 60.0,
    max_bytes: int = _MAX_URL_BYTES,
    max_redirects: int = _MAX_REDIRECTS,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> tuple[bytes, str, str]:
    """GET a public URL. Each redirect is checked and pinned before connect.

    Returns ``(body, content_type, final_url)``. Closes an owned client on error.
    Does not follow redirects inside httpx.
    """
    own = client is None
    if client is None:
        client = httpx.Client(
            follow_redirects=False,
            timeout=timeout,
            trust_env=False,
        )
    current = url
    seen: set[str] = set()
    try:
        for hop in range(max_redirects + 1):
            if current in seen:
                raise UnsafeUrlError("redirect loop")
            seen.add(current)
            host, ip = public_connect_target(current)
            with _pin_getaddrinfo(host, ip):
                with client.stream(
                    "GET",
                    current,
                    headers=headers,
                    follow_redirects=False,
                ) as resp:
                    if resp.status_code in _REDIRECTS:
                        if hop == max_redirects:
                            raise UnsafeUrlError("too many redirects")
                        loc = resp.headers.get("location")
                        if not loc:
                            raise UnsafeUrlError("redirect without location")
                        current = urljoin(str(resp.url), loc)
                        continue
                    if resp.status_code != 200:
                        # Don't drain the body: a 202 challenge can hold the
                        # socket open and the approval request never returns.
                        code = resp.status_code
                        request = resp.request
                        resp.close()
                        raise httpx.HTTPStatusError(
                            f"HTTP {code}",
                            request=request,
                            response=resp,
                        )
                    body = read_capped(resp.iter_bytes(chunk_size=65536), max_bytes)
                    ctype = resp.headers.get("content-type", "")
                    return body, ctype, str(resp.url)
        raise UnsafeUrlError("too many redirects")
    finally:
        if own:
            client.close()

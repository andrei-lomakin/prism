"""File / URL parsers → plain text (+ light structure hints)."""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            t = page.extract_text() or ""
            if t.strip():
                pages.append(f"<!-- page:{i} -->\n{t}")
        text = "\n\n".join(pages)
        logger.info("PDF parsed: %s pages=%s chars=%s", path.name, len(reader.pages), len(text))
        return text

    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        logger.info("DOCX parsed: %s chars=%s", path.name, len(text))
        return text

    if suffix in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return html_to_text(raw)

    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.info("Text file parsed: %s chars=%s", path.name, len(text))
    return text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


_MAX_REDIRECTS = 5
_MAX_URL_BYTES = 20_000_000
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


def _assert_host_public(host: str) -> None:
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
        return
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


def assert_public_http_url(url: str) -> str:
    """Reject non-http(s) URLs and loopback / private / link-local targets."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError(f"unsupported scheme {parsed.scheme or '(none)'}")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("credentials in URL are not allowed")
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError("missing host")
    _assert_host_public(host)
    return url


def download_url_to_file(url: str, dest: Path, *, client: httpx.Client | None = None) -> Path:
    """Fetch a public URL. Each redirect is checked before it is followed."""
    if client is None:
        with httpx.Client(follow_redirects=False, timeout=60.0) as owned:
            return _download_checked(owned, url, dest)
    return _download_checked(client, url, dest)


def _download_checked(client: httpx.Client, url: str, dest: Path) -> Path:
    current = url
    seen: set[str] = set()
    for _ in range(_MAX_REDIRECTS + 1):
        if current in seen:
            raise UnsafeUrlError("redirect loop")
        seen.add(current)
        assert_public_http_url(current)
        resp = client.get(current)
        if resp.status_code in {301, 302, 303, 307, 308}:
            loc = resp.headers.get("location")
            if not loc:
                raise UnsafeUrlError("redirect without location")
            current = urljoin(current, loc)
            continue
        resp.raise_for_status()
        if len(resp.content) > _MAX_URL_BYTES:
            raise UnsafeUrlError(f"response exceeds {_MAX_URL_BYTES} bytes")
        content_type = resp.headers.get("content-type", "").lower()
        body = resp.text
        looks_html = (
            "html" in content_type
            or current.rstrip("/").endswith((".html", ".htm"))
            or "<html" in body[:500].lower()
        )
        dest = dest.with_suffix(".txt")
        dest.write_text(html_to_text(body) if looks_html else body, encoding="utf-8")
        return dest
    raise UnsafeUrlError("too many redirects")


def suggest_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = (parsed.path.rstrip("/").split("/")[-1] or parsed.netloc or "url_document").strip()
    return (name or "url_document")[:200]


def content_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_filename(name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    clean = re.sub(r"[^\w.\-]+", "_", stem)[:80] or "upload"
    return f"{clean}{suffix}"

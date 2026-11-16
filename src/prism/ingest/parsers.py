"""File / URL parsers → plain text (+ light structure hints)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

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


def download_url_to_file(url: str, dest: Path) -> Path:
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        body = resp.text

    looks_html = (
        "html" in content_type
        or url.rstrip("/").endswith((".html", ".htm"))
        or "<html" in body[:500].lower()
    )
    dest = dest.with_suffix(".txt")
    dest.write_text(html_to_text(body) if looks_html else body, encoding="utf-8")
    return dest


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

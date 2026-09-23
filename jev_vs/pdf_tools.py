"""Thin helpers over liteparse (pdfium + tesseract) for the document tasks.

Everything jev sees has to be text, so this module is the bridge: it turns
PDF pages into text (native or OCR), renders pages to images, and produces the
"four-way OCR" candidates used for orientation detection.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from liteparse import LiteParse
from PIL import Image

ANGLES = (0, 90, 180, 270)


@dataclass
class PageText:
    page_num: int
    text: str
    needs_ocr: bool = False
    reasons: list[str] = field(default_factory=list)
    ocr_confidence: float | None = None  # mean tesseract confidence when OCR was used


@lru_cache(maxsize=4)
def _parser(ocr: bool, ocr_language: str, dpi: float) -> LiteParse:
    return LiteParse(ocr_enabled=ocr, ocr_language=ocr_language, dpi=dpi, output_format="text", quiet=True)


def pdf_info(path: str | Path) -> dict:
    """Cheap document summary: page count, per-page OCR verdicts, producer."""
    parser = _parser(False, "eng", 100)
    stats = parser.is_complex(str(path))
    result = parser.parse(str(path))
    return {
        "path": str(path),
        "pages": result.total_pages,
        "producer": result.producer,
        "creator": result.creator,
        "pages_needing_ocr": [s.page_number for s in stats if s.needs_ocr],
        "reasons": {s.page_number: s.reasons for s in stats if s.reasons},
        "native_chars": sum(len(p.text) for p in result.pages),
    }


def page_texts(source: str | Path | bytes, *, ocr: bool = True, ocr_language: str = "eng", dpi: float = 200) -> list[PageText]:
    """Per-page text. liteparse only OCRs pages/images that have no native text."""
    parser = _parser(ocr, ocr_language, dpi)
    src = source if isinstance(source, bytes) else str(source)
    verdicts = {s.page_number: s for s in parser.is_complex(src)}
    result = parser.parse(src)
    out = []
    for page in result.pages:
        v = verdicts.get(page.page_num)
        confs = [t.confidence for t in page.text_items if t.confidence is not None]
        out.append(
            PageText(
                page_num=page.page_num,
                text=page.text.strip(),
                needs_ocr=bool(v and v.needs_ocr),
                reasons=list(v.reasons) if v else [],
                ocr_confidence=(sum(confs) / len(confs)) if confs else None,
            )
        )
    return out


def render_pages(path: str | Path, page_numbers: Iterable[int] | None = None, *, dpi: float = 100) -> list[tuple[int, Image.Image]]:
    """Render pages to PIL images via pdfium."""
    parser = _parser(False, "eng", dpi)
    shots = parser.screenshot(str(path), page_numbers=list(page_numbers) if page_numbers else None)
    return [(s.page_num, Image.open(io.BytesIO(s.image_bytes)).convert("RGB")) for s in shots]


def image_to_pdf_bytes(img: Image.Image, dpi: float = 150) -> bytes:
    """Wrap an image in a one-page, image-only PDF (a 'scan'). No text layer."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PDF", resolution=dpi)
    return buf.getvalue()


def ocr_image(img: Image.Image, *, language: str = "eng", dpi: float = 150) -> PageText:
    """OCR a PIL image by routing it through liteparse as an image-only PDF."""
    pages = page_texts(image_to_pdf_bytes(img, dpi), ocr=True, ocr_language=language, dpi=dpi)
    return pages[0] if pages else PageText(1, "")


def rotation_candidates(img: Image.Image, *, language: str = "eng", angles: Iterable[int] = ANGLES) -> dict[int, PageText]:
    """OCR the image after rotating it counter-clockwise by each candidate angle.

    A page scanned `theta` degrees clockwise reads correctly for candidate == theta.
    """
    return {a: ocr_image(img.rotate(a, expand=True, fillcolor="white"), language=language) for a in angles}


def head(text: str, n: int) -> str:
    return text[:n]


def tail(text: str, n: int) -> str:
    return text[-n:] if len(text) > n else text

"""Build small, labelled document corpora with known ground truth.

Everything ends up under ./data as real PDFs plus a `manifest.jsonl` per task, so the
notebook (and the CLI scripts) only ever deal with files on disk:

  data/language/     one-page native-text PDFs, one Wikipedia excerpt each, 12 languages
  data/orientation/  image-only "scans" of those pages, rotated 0/90/180/270 clockwise
  data/classify/     RVL-CDIP scans (16 document classes) saved as image-only PDFs
  data/split/        bundles of several multi-page Wikipedia articles concatenated

Rendering uses reportlab; scanning uses pdfium via liteparse; OCR is tesseract via liteparse.
"""

from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path

from PIL import Image

from . import pdf_tools

DATA = Path(os.environ.get("JEV_VS_DATA", "data"))

LANGS = ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "tr", "el", "vi"]
LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "pl": "Polish", "ru": "Russian", "tr": "Turkish", "el": "Greek", "vi": "Vietnamese",
}

RVL_CDIP_LABELS = [
    "advertisement", "budget", "email", "file_folder", "form", "handwritten", "invoice", "letter",
    "memo", "news_article", "presentation", "questionnaire", "resume", "scientific_publication",
    "scientific_report", "specification",
]

FONT_CANDIDATES = [
    os.environ.get("JEV_VS_FONT", ""),
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


# ------------------------------------------------------------------ rendering
def _font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for p in FONT_CANDIDATES:
        if p and Path(p).exists():
            if "DocFont" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("DocFont", p))
            return "DocFont"
    return "Helvetica"  # Latin-1 only


def render_text_pdf(path: Path, title: str, body: str, *, font_size: int = 11) -> int:
    """Render a title + paragraphs to a paginated PDF. Returns the page count."""
    from pypdf import PdfReader
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    font = _font()
    body_style = ParagraphStyle("body", fontName=font, fontSize=font_size, leading=font_size * 1.4, spaceAfter=8)
    title_style = ParagraphStyle("title", fontName=font, fontSize=font_size + 9, leading=font_size * 2.2, spaceAfter=14)
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch, title=title)
    flow = [Paragraph(escape(title), title_style), Spacer(1, 6)]
    for para in (p.strip() for p in body.split("\n")):
        if para:
            flow.append(Paragraph(escape(para), body_style))
    doc.build(flow)
    return len(PdfReader(str(path)).pages)


def _write_manifest(dir_: Path, rows: list[dict]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / "manifest.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def load_manifest(task: str) -> list[dict]:
    p = DATA / task / "manifest.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `uv run build-corpus {task}` first")
    return [json.loads(l) for l in open(p)]


# ------------------------------------------------------------------ wikipedia
def _wiki(lang: str, min_chars: int, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    for row in ds:
        text = row["text"]
        if len(text) < min_chars or "\n" not in text:
            continue
        if any(k in row["title"].lower() for k in ("list of", "disambiguation")):
            continue
        yield row["title"], text


def _first_chars(text: str, n: int) -> str:
    out, total = [], 0
    for para in text.split("\n"):
        para = para.strip()
        if not para or len(para) < 40:
            continue  # drop section headings and stubs
        out.append(para)
        total += len(para)
        if total >= n:
            break
    return "\n".join(out)


# ------------------------------------------------------------------ builders
def build_language(n_per_lang: int = 5, langs: list[str] = LANGS, chars: int = 1400, seed: int = 0) -> Path:
    d = DATA / "language"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang in langs:
        for i, (title, text) in zip(range(n_per_lang), _wiki(lang, chars + 500, seed)):
            body = _first_chars(text, chars)
            path = d / f"{lang}_{i:02d}.pdf"
            render_text_pdf(path, title, body)
            rows.append({"id": f"lang-{lang}-{i:02d}", "path": str(path), "lang": lang, "title": title, "text": body})
            print(f"[language] {lang} {i}: {title}")
    return _write_manifest(d, rows)


def build_orientation(n: int = 40, seed: int = 0, dpi: int = 150) -> Path:
    """Rotate rendered language pages clockwise and save them as image-only 'scans'."""
    src = load_manifest("language")
    rng = random.Random(seed)
    rng.shuffle(src)
    d = DATA / "orientation"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(src[:n]):
        (_, img), = pdf_tools.render_pages(row["path"], [1], dpi=dpi)
        theta = rng.choice(pdf_tools.ANGLES)
        scan = img.rotate(-theta, expand=True, fillcolor="white")  # PIL rotates CCW; we want CW by theta
        path = d / f"scan_{i:03d}.pdf"
        path.write_bytes(pdf_tools.image_to_pdf_bytes(scan, dpi))
        rows.append({"id": f"orient-{i:03d}", "path": str(path), "rotation_cw": theta, "source_id": row["id"], "lang": row["lang"]})
        print(f"[orientation] {i}: {row['lang']} rotated {theta}° cw")
    return _write_manifest(d, rows)


def build_classify(n_per_class: int = 6, seed: int = 0, ocr_language: str = "eng") -> Path:
    """Pull RVL-CDIP scans (chainyo/rvl-cdip, test split) and OCR them with liteparse."""
    import io as _io

    from datasets import Image as HFImage, load_dataset

    d = DATA / "classify"
    d.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("chainyo/rvl-cdip", split="test", streaming=True).shuffle(seed=seed, buffer_size=500)
    ds = ds.cast_column("image", HFImage(decode=False))  # decode ourselves: a few TIFFs are broken
    counts = {k: 0 for k in range(16)}
    rows = []
    for row in ds:
        label = int(row["label"])
        if counts[label] >= n_per_class:
            if all(c >= n_per_class for c in counts.values()):
                break
            continue
        try:
            img: Image.Image = Image.open(_io.BytesIO(row["image"]["bytes"])).convert("RGB")
        except Exception as e:  # a few RVL-CDIP TIFFs are undecodable; skip them
            print(f"[classify] skipping undecodable image: {e}")
            continue
        i = counts[label]
        counts[label] += 1
        path = d / f"{RVL_CDIP_LABELS[label]}_{i:02d}.pdf"
        path.write_bytes(pdf_tools.image_to_pdf_bytes(img, 200))
        pages = pdf_tools.page_texts(path, ocr=True, ocr_language=ocr_language)
        text = pages[0].text if pages else ""
        rows.append({"id": f"cls-{RVL_CDIP_LABELS[label]}-{i:02d}", "path": str(path), "label": RVL_CDIP_LABELS[label], "text": text, "ocr_confidence": pages[0].ocr_confidence if pages else None})
        print(f"[classify] {RVL_CDIP_LABELS[label]} {i}: {len(text)} chars OCR")
    return _write_manifest(d, rows)


def build_split(n_bundles: int = 8, docs_per_bundle: tuple[int, int] = (3, 5), seed: int = 0, lang: str = "en") -> Path:
    """Render multi-page articles and concatenate them into bundles with known boundaries."""
    from pypdf import PdfWriter

    d = DATA / "split"
    (d / "docs").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    wiki = _wiki(lang, 6000, seed)
    rows = []
    for b in range(n_bundles):
        k = rng.randint(*docs_per_bundle)
        writer = PdfWriter()
        starts, titles, page_texts, page = [], [], [], 1
        for j in range(k):
            title, text = next(wiki)
            body = _first_chars(text, rng.randint(3500, 9000))
            doc_path = d / "docs" / f"b{b:02d}_d{j}.pdf"
            n_pages = render_text_pdf(doc_path, title, body)
            writer.append(str(doc_path))
            starts.append(page)
            titles.append(title)
            page += n_pages
        path = d / f"bundle_{b:02d}.pdf"
        with open(path, "wb") as f:
            writer.write(f)
        texts = [p.text for p in pdf_tools.page_texts(path, ocr=False)]
        rows.append({"id": f"split-{b:02d}", "path": str(path), "n_pages": page - 1, "doc_starts": starts, "titles": titles, "page_texts": texts})
        print(f"[split] bundle {b}: {k} docs, {page - 1} pages, starts={starts}")
    return _write_manifest(d, rows)


BUILDERS = {"language": build_language, "orientation": build_orientation, "classify": build_classify, "split": build_split}


# ------------------------------------------------------------------ triage / fan-out
TESS_LANG = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "nl": "nld", "pl": "pol", "ru": "rus", "tr": "tur", "el": "ell", "vi": "vie"}

DEGRADATIONS = {
    0: "clean 150 dpi scan",
    1: "mild blur",
    2: "low resolution (72 dpi) + blur",
    3: "low resolution + blur + noise + skew",
}


def degrade(img: Image.Image, level: int, rng: random.Random) -> Image.Image:
    """Simulate progressively worse scans. Level 0 returns the input unchanged."""
    from PIL import ImageFilter

    if level == 0:
        return img
    out = img
    if level >= 2:
        w, h = out.size
        out = out.resize((int(w * 0.48), int(h * 0.48)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    out = out.filter(ImageFilter.GaussianBlur(radius=0.9 if level == 1 else 1.4))
    if level >= 3:
        import numpy as np

        arr = np.asarray(out).astype("int16")
        arr = arr + rng.choice([-1, 1]) * np.random.default_rng(rng.randint(0, 2**31)).integers(0, 70, arr.shape, dtype="int16")
        out = Image.fromarray(arr.clip(0, 255).astype("uint8"))
        out = out.rotate(rng.uniform(-3, 3), expand=False, fillcolor="white")
    return out


def _cer(hyp: str, ref: str) -> float:
    from rapidfuzz.distance import Levenshtein

    norm = lambda s: " ".join(s.split())
    return Levenshtein.normalized_distance(norm(hyp), norm(ref))


def build_triage(n: int = 48, seed: int = 1, dpi: int = 150, cer_threshold: float = 0.10) -> Path:
    """Degraded + rotated scans of language pages, with everything jev needs pre-computed.

    Ground truth per page: language, rotation, and `upgrade` = the local OCR (at the correct
    rotation) has a character error rate above `cer_threshold` against the known source text.
    """
    src = load_manifest("language")
    rng = random.Random(seed)
    rng.shuffle(src)
    d = DATA / "triage"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(src[:n]):
        (_, img), = pdf_tools.render_pages(row["path"], [1], dpi=dpi)
        level = rng.choice([0, 0, 1, 2, 3])
        theta = rng.choice(pdf_tools.ANGLES)
        scan = degrade(img, level, rng).rotate(-theta, expand=True, fillcolor="white")
        path = d / f"scan_{i:03d}.pdf"
        path.write_bytes(pdf_tools.image_to_pdf_bytes(scan, dpi))
        lang = TESS_LANG[row["lang"]]
        cands = pdf_tools.rotation_candidates(scan, language=lang)
        upright = cands[theta]
        stats = pdf_tools._parser(True, lang, dpi).is_complex(str(path))[0]
        cer = _cer(upright.text, row["title"] + "\n" + row["text"])
        rows.append(
            {
                "id": f"triage-{i:03d}",
                "path": str(path),
                "lang": row["lang"],
                "rotation_cw": theta,
                "degradation": level,
                "cer": round(cer, 4),
                "upgrade": cer > cer_threshold,
                "ocr_confidence": upright.ocr_confidence,
                "candidates": {f"rotation_{a}": pt.text[:1500] for a, pt in cands.items()},
                "complexity": {
                    "needs_ocr": stats.needs_ocr, "reasons": stats.reasons, "text_coverage": round(stats.text_coverage, 3),
                    "full_page_image": stats.full_page_image, "is_garbled": stats.is_garbled, "native_text_length": stats.text_length,
                },
            }
        )
        print(f"[triage] {i}: {row['lang']} rot={theta} level={level} cer={cer:.3f} conf={upright.ocr_confidence} -> {'upgrade' if cer > cer_threshold else 'local_ok'}")
    return _write_manifest(d, rows)


BUILDERS["triage"] = build_triage

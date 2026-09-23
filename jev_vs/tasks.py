"""Turn corpus manifests into backend-agnostic `Decision`s, one per document task."""

from __future__ import annotations

import json
from pathlib import Path

from . import pdf_tools
from .corpus import DATA, LANG_NAMES, RVL_CDIP_LABELS, load_manifest
from .decisions import Decision

# ------------------------------------------------------------------ language
def language_decisions(langs: list[str] | None = None, max_chars: int = 1200) -> list[Decision]:
    rows = load_manifest("language")
    langs = langs or sorted({r["lang"] for r in rows})
    options = {l: f"The text is written in {LANG_NAMES[l]}." for l in langs}
    out = []
    for r in rows:
        if r["lang"] not in langs:
            continue
        # Re-extract from the PDF so the demo really goes PDF -> liteparse -> decision.
        text = pdf_tools.page_texts(r["path"], ocr=False)[0].text[:max_chars]
        out.append(Decision(r["id"], text, "Which language is this document written in?", options, truth=r["lang"], meta={"path": r["path"]}))
    return out


# ------------------------------------------------------------------ orientation
ORIENTATION_OPTIONS = {
    "0": "Candidate `rotation_0` is coherent, correctly oriented natural-language text.",
    "90": "Candidate `rotation_90` is coherent, correctly oriented natural-language text.",
    "180": "Candidate `rotation_180` is coherent, correctly oriented natural-language text.",
    "270": "Candidate `rotation_270` is coherent, correctly oriented natural-language text.",
}
ORIENTATION_QUESTION = (
    "A scanned page was OCR'd four times, after rotating the image by 0, 90, 180 and 270 degrees. "
    "Only the correct rotation yields readable text; the others produce fragments and garbage. "
    "Which candidate is the readable one?"
)


def orientation_candidates(max_chars: int = 500, cache: Path | None = None) -> dict[str, dict[str, str]]:
    """OCR every scan 4 ways (cached: this is the slow, tesseract-bound step)."""
    cache = cache or DATA / "orientation" / "candidates.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for r in load_manifest("orientation"):
        (_, img), = pdf_tools.render_pages(r["path"], [1], dpi=150)
        cands = pdf_tools.rotation_candidates(img)
        out[r["id"]] = {f"rotation_{a}": pt.text[:max_chars] for a, pt in cands.items()}
        print(f"[ocr x4] {r['id']} truth={r['rotation_cw']} chars={[len(v) for v in out[r['id']].values()]}")
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


def orientation_decisions() -> list[Decision]:
    cands = orientation_candidates()
    return [
        Decision(r["id"], cands[r["id"]], ORIENTATION_QUESTION, ORIENTATION_OPTIONS, truth=str(r["rotation_cw"]), meta={"path": r["path"], "lang": r["lang"]})
        for r in load_manifest("orientation")
    ]


# ------------------------------------------------------------------ classification
CLASS_DESCRIPTIONS = {
    "advertisement": "Marketing or promotional material for a product, brand or event.",
    "budget": "Financial planning tables: budgets, cost breakdowns, allocations by line item.",
    "email": "An email message with To/From/Subject/Date headers.",
    "file_folder": "A file folder label or tab; almost no content beyond a title or code.",
    "form": "A blank or filled-in form with labelled fields and boxes.",
    "handwritten": "Predominantly handwritten notes or a handwritten letter.",
    "invoice": "An invoice, bill or purchase order listing items, quantities and amounts due.",
    "letter": "A typed business or personal letter with salutation and signature.",
    "memo": "An internal memorandum with MEMO/TO/FROM/RE style header.",
    "news_article": "A newspaper or magazine article with headline and columns of prose.",
    "presentation": "Slides or overhead transparencies: large text, bullets, one idea per page.",
    "questionnaire": "A survey or questionnaire with numbered questions and answer options.",
    "resume": "A CV or résumé: personal details, education, work history.",
    "scientific_publication": "A journal article or conference paper with abstract, sections, references.",
    "scientific_report": "A technical or laboratory report: methods, results, tables, not a journal paper.",
    "specification": "A product or technical specification sheet with parameters and tolerances.",
}
CLASSIFY_QUESTION = "The state is OCR text from one scanned page. What kind of document is it?"


def classify_decisions(max_chars: int = 2500) -> list[Decision]:
    return [
        Decision(r["id"], r["text"][:max_chars] or "(no OCR text)", CLASSIFY_QUESTION, CLASS_DESCRIPTIONS, truth=r["label"], meta={"path": r["path"], "ocr_confidence": r.get("ocr_confidence")})
        for r in load_manifest("classify")
    ]


# ------------------------------------------------------------------ splitting
SPLIT_OPTIONS = {
    "new_document": "The current page starts a new, separate document (new title, new subject, no continuity with the previous page).",
    "continuation": "The current page continues the same document as the previous page.",
}
SPLIT_QUESTION = "You see the end of the previous page and the start of the current page of a scanned bundle. Does the current page begin a new document?"


def split_decisions(context_chars: int = 700) -> list[Decision]:
    out = []
    for r in load_manifest("split"):
        texts = r["page_texts"]
        starts = set(r["doc_starts"])
        for p in range(2, r["n_pages"] + 1):
            state = {"previous_page_end": pdf_tools.tail(texts[p - 2], context_chars), "current_page_start": pdf_tools.head(texts[p - 1], context_chars)}
            out.append(Decision(f"{r['id']}-p{p:02d}", state, SPLIT_QUESTION, SPLIT_OPTIONS, truth="new_document" if p in starts else "continuation", meta={"bundle": r["id"], "page": p}))
    return out


TASKS = {"language": language_decisions, "orientation": orientation_decisions, "classify": classify_decisions, "split": split_decisions}


# ------------------------------------------------------------------ parse-tier triage
TRIAGE_OPTIONS = {
    "local_ok": "The local parse captured the page faithfully: fluent text, few OCR artefacts. Use it as is.",
    "upgrade": "The local parse is unreliable (garbled words, missing text, low OCR confidence, or no text at all). Send the page to a higher-tier parser.",
}
TRIAGE_QUESTION = (
    "The state holds liteparse's cheap complexity signals for one scanned page plus an excerpt of its local OCR output and the OCR engine's mean confidence. "
    "Is the local parse good enough to use, or should this page be escalated to a slower, higher-accuracy parser?"
)


def _triage_state(r: dict, excerpt_chars: int = 900) -> dict:
    upright = r["candidates"][f"rotation_{r['rotation_cw']}"]
    return {
        "complexity": r["complexity"],
        "ocr": {"mean_confidence": round(r["ocr_confidence"], 3) if r["ocr_confidence"] is not None else None, "chars": len(upright), "excerpt": upright[:excerpt_chars]},
    }


def triage_decisions() -> list[Decision]:
    """Assumes orientation was fixed upstream: the OCR excerpt is the correctly rotated candidate."""
    return [
        Decision(r["id"], _triage_state(r), TRIAGE_QUESTION, TRIAGE_OPTIONS, truth="upgrade" if r["upgrade"] else "local_ok", meta={"cer": r["cer"], "degradation": r["degradation"], "lang": r["lang"]})
        for r in load_manifest("triage")
    ]


# ------------------------------------------------------------------ fan-out
FANOUT_DOC_TYPES = {**CLASS_DESCRIPTIONS, "encyclopedia_article": "A reference / encyclopedia entry: a titled topic followed by neutral explanatory prose."}


def fanout_items(langs: list[str] | None = None, cand_chars: int = 500):
    """One state per scanned page, four questions: orientation, language, document type, parse-tier triage."""
    from .decisions import FanoutItem

    rows = load_manifest("triage")
    langs = langs or sorted(LANG_NAMES)
    lang_options = {l: f"The text is written in {LANG_NAMES[l]}." for l in langs}
    items = []
    for r in rows:
        state = {"ocr_candidates": {k: v[:cand_chars] for k, v in r["candidates"].items()}, **_triage_state(r)}
        items.append(
            FanoutItem(
                r["id"],
                state,
                {
                    "orientation": (ORIENTATION_QUESTION.replace("A scanned page", "The page in `ocr_candidates`"), ORIENTATION_OPTIONS, str(r["rotation_cw"])),
                    "language": ("Which language is the page written in? (Judge from the readable candidate.)", lang_options, r["lang"]),
                    "doc_type": ("What kind of document is this page?", FANOUT_DOC_TYPES, "encyclopedia_article"),
                    "triage": (TRIAGE_QUESTION.replace("The state holds", "`complexity` and `ocr` hold"), TRIAGE_OPTIONS, "upgrade" if r["upgrade"] else "local_ok"),
                },
                meta={"lang": r["lang"], "degradation": r["degradation"]},
            )
        )
    return items


TASKS["triage"] = triage_decisions

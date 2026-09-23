"""pdf-text FILE [--no-ocr] [--lang eng] [--json] : per-page text, OCR only where needed."""
import argparse
import json
from dataclasses import asdict

from jev_vs import pdf_tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--lang", default="eng", help="tesseract language code(s), e.g. eng+deu")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    pages = pdf_tools.page_texts(a.file, ocr=not a.no_ocr, ocr_language=a.lang)
    if a.json:
        print(json.dumps([asdict(p) for p in pages], ensure_ascii=False, indent=2))
        return
    for p in pages:
        tag = f" [OCR {p.ocr_confidence:.2f}]" if p.ocr_confidence is not None else ""
        print(f"===== page {p.page_num}{tag} {'needs_ocr:' + ','.join(p.reasons) if p.needs_ocr else ''}\n{p.text}\n")


if __name__ == "__main__":
    main()

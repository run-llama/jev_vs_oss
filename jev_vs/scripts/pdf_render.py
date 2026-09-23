"""pdf-render FILE [-o DIR] [--pages 1,3] [--dpi 100] [--rotate 90] : render pages to PNG (optionally rotated)."""
import argparse
from pathlib import Path

from jev_vs import pdf_tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-o", "--out", default="renders")
    ap.add_argument("--pages", default=None, help="comma separated 1-indexed page numbers")
    ap.add_argument("--dpi", type=float, default=100)
    ap.add_argument("--rotate", type=int, default=0, help="clockwise degrees to rotate the output")
    ap.add_argument("--as-pdf", action="store_true", help="write an image-only PDF ('scan') instead of PNGs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = [int(x) for x in a.pages.split(",")] if a.pages else None
    for num, img in pdf_tools.render_pages(a.file, pages, dpi=a.dpi):
        if a.rotate:
            img = img.rotate(-a.rotate, expand=True, fillcolor="white")
        stem = f"{Path(a.file).stem}_p{num}"
        if a.as_pdf:
            (out / f"{stem}.pdf").write_bytes(pdf_tools.image_to_pdf_bytes(img, a.dpi))
        else:
            img.save(out / f"{stem}.png")
        print(out / stem)


if __name__ == "__main__":
    main()

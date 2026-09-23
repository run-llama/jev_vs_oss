"""pdf-orient FILE [--page 1] : OCR a page at 0/90/180/270 and print the four candidates + tesseract OSD."""
import argparse

from jev_vs import pdf_tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--chars", type=int, default=300)
    a = ap.parse_args()
    (_, img), = pdf_tools.render_pages(a.file, [a.page], dpi=150)
    for angle, pt in pdf_tools.rotation_candidates(img).items():
        conf = f"{pt.ocr_confidence:.2f}" if pt.ocr_confidence is not None else "n/a"
        print(f"===== rotate {angle}° ccw  (chars={len(pt.text)}, ocr_conf={conf})\n{pt.text[: a.chars]!r}\n")
    try:
        import pytesseract

        print("===== tesseract OSD\n" + pytesseract.image_to_osd(img, config="--psm 0"))
    except Exception as e:
        print("tesseract OSD failed:", e)


if __name__ == "__main__":
    main()

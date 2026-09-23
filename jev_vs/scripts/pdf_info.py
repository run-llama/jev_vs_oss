"""pdf-info FILE... : page count, producer, which pages need OCR (liteparse is_complex)."""
import argparse
import json

from jev_vs import pdf_tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    for f in ap.parse_args().files:
        print(json.dumps(pdf_tools.pdf_info(f), indent=2))


if __name__ == "__main__":
    main()

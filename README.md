# jev vs. open alternatives — document tasks

Material comparing [jev](https://docs.typesafe.ai/introduction/quickstart) (TypeSafe AI's hosted
decision model) with open alternatives on four document-pipeline chores, with
[liteparse](https://github.com/run-llama/liteparse) (pdfium + tesseract) doing all the PDF work.

| Task | Data (generated, labelled) | jev | Open decoder on Modal | Open encoders (local) | Specialised open tool |
|---|---|---|---|---|---|
| Language detection | Wikipedia excerpts rendered to PDF, 12 languages | Choice(12) | Qwen3.5-4B, SemIf-style direct logits | Laya, jeff | lingua |
| Orientation detection | those pages "scanned" and rotated 0/90/180/270° | Choice(4) over 4-way OCR text | same | same | tesseract OSD |
| Document classification | RVL-CDIP scans, 16 classes, OCR'd | Choice(16) | same | same | — |
| Bundle splitting | multi-page articles concatenated | Choice(2) per page boundary | same | same | — |
| Parse-tier triage | degraded + rotated scans, labelled by OCR character error rate vs. source | Choice(2) on liteparse complexity signals + OCR excerpt | same | same | confidence/length heuristic |

## Quick start

```bash
uv sync
cp .env.example .env                 # add TYPESAFE_API_KEY
uv run build-corpus all              # downloads samples, renders + OCRs PDFs into ./data (~5-10 min)
uv run modal deploy modal_app.py     # serves Qwen/Qwen3.5-4B on an L4; `modal run modal_app.py` smoke-tests it
uv run jupyter lab jev_vs_open.ipynb
```

`build-corpus <task> --n N` rebuilds one corpus at a different size.

Or, just read [the notebook](jev_vs_open.ipynb) to follow along.

## liteparse CLIs

Small "binaries" over `jev_vs/pdf_tools.py`, installed by `uv sync`:

```bash
uv run pdf-info   doc.pdf                 # pages, producer, which pages need OCR and why
uv run pdf-text   doc.pdf [--lang deu]    # per-page text; OCR only where there is no native text
uv run pdf-render doc.pdf --pages 1 --rotate 90 --as-pdf   # rasterise pages (optionally into a fake scan)
uv run pdf-orient scan.pdf                # OCR at 0/90/180/270 + tesseract OSD, side by side
```

## Links

* LiteParse: https://github.com/run-llama/liteparse · https://developers.llamaindex.ai/liteparse/
* LlamaParse: https://developers.llamaindex.ai/llamaparse/
* jev docs: https://docs.typesafe.ai — [models & pricing](https://docs.typesafe.ai/models),
  [confidence](https://docs.typesafe.ai/confidence), [jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
* JevBench: https://benchmarkheaven.com/jev-models
* SemIf / OpenJev (open baselines): https://github.com/TheoLeeCJ/SemIf, https://huggingface.co/AlexWortega/openjev

## Open encoders

* [Laya](https://huggingface.co/convaiinnovations/laya): 421M ModernBERT + decision head, jev-compatible `state`/`questions`
  schema with native fan-out. Runs locally (MPS/CPU). We use the `typed-decisions` checkpoint with the card's
  budget knobs raised (`max_len=2048`, `head_max_len=512`) so full states and 17-option questions fit.
* [jeff](https://github.com/logan-markewich/jeff): a self-hosted server implementing the jev System One API on
  gliformer-large-v1 (400M). Because it speaks the same wire format, the notebook drives it with `JevBackend`
  and just a `base_url`. Run it locally (`JEFF_API_KEYS=devkey uv run jeff` in the jeff repo, then
  `JEFF_BASE_URL=http://localhost:8000`) or on Modal (`deploy/modal_gpu.py`). Set `JEFF_BASE_URL` / `JEFF_API_KEY` in `.env`.

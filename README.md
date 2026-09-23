# jev vs. open alternatives — document tasks

Webinar material comparing [jev](https://docs.typesafe.ai/introduction/quickstart) (TypeSafe AI's hosted
decision model) with open alternatives on four document-pipeline chores, with
[liteparse](https://github.com/run-llama/liteparse) (pdfium + tesseract) doing all the PDF work.

| Task | Data (generated, labelled) | jev | Open decoder on Modal | Open encoders (local) | Specialised open tool |
|---|---|---|---|---|---|
| Language detection | Wikipedia excerpts rendered to PDF, 12 languages | Choice(12) | Qwen3.5-4B, SemIf-style direct logits | Laya, jeff | lingua |
| Orientation detection | those pages "scanned" and rotated 0/90/180/270° | Choice(4) over 4-way OCR text | same | same | tesseract OSD |
| Document classification | RVL-CDIP scans, 16 classes, OCR'd | Choice(16) | same | same | — |
| Bundle splitting | multi-page articles concatenated | Choice(2) per page boundary | same | same | — |
| Parse-tier triage | degraded + rotated scans, labelled by OCR character error rate vs. source | Choice(2) on liteparse complexity signals + OCR excerpt | same | same | confidence/length heuristic |
| Fan-out | the triage scans | orientation + language + doc type + triage in **one** call | 4 forwards | Laya, jeff: 1 call | — |

## Quick start

```bash
uv sync
cp .env.example .env                 # add TYPESAFE_API_KEY
uv run build-corpus all              # downloads samples, renders + OCRs PDFs into ./data (~5-10 min)
uv run modal deploy modal_app.py     # serves Qwen/Qwen3.5-4B on an L4; `modal run modal_app.py` smoke-tests it
uv run jupyter lab jev_vs_open.ipynb
```

`build-corpus <task> --n N` rebuilds one corpus at a different size.

## liteparse CLIs

Small "binaries" over `jev_vs/pdf_tools.py`, installed by `uv sync`:

```bash
uv run pdf-info   doc.pdf                 # pages, producer, which pages need OCR and why
uv run pdf-text   doc.pdf [--lang deu]    # per-page text; OCR only where there is no native text
uv run pdf-render doc.pdf --pages 1 --rotate 90 --as-pdf   # rasterise pages (optionally into a fake scan)
uv run pdf-orient scan.pdf                # OCR at 0/90/180/270 + tesseract OSD, side by side
```

## Layout

```
jev_vs/
  pdf_tools.py     liteparse wrappers: page text (native/OCR), rendering, image->PDF, 4-way OCR
  corpus.py        builds ./data/{language,orientation,classify,split}/manifest.jsonl with ground truth
  tasks.py         manifests -> backend-agnostic Decision(state, question, options, truth)
  decisions.py     Decision/DecisionResult + JevBackend (typesafe-sdk) + OpenLogitsBackend (Modal) + fan-out helpers
  baselines.py     lingua, tesseract OSD, confidence heuristic, Laya (local encoder), gliformer-base (naive, unused by default)
  evalh.py         comparison tables, confusion matrices, charts
  scripts/         the CLIs above
modal_app.py       Modal class serving any HF causal LM with the SemIf direct-logits readout
jev_vs_open.ipynb  The notebook with all the experiments and charts
```

## How the comparison is kept fair

* Both models receive the identical `state` / question / option descriptions. jev gets a `Choice`
  question; the open model gets the SemIf prompt (system + JSON evidence/criterion/options) and we
  read the next-token logits over the option letters — no generation, no JSON parsing.
* jev cost = reported input tokens × $0.042/M. Open-model cost = wall seconds per decision × Modal
  L4 price. Latency for the open model is batched and warm; JevBench numbers are single-request.
* jev is text-only, so orientation is turned into a text question: OCR the page at four rotations
  and ask which candidate is readable.

## Links

* jev docs: https://docs.typesafe.ai — [models & pricing](https://docs.typesafe.ai/models),
  [confidence](https://docs.typesafe.ai/confidence), [jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
* JevBench: https://benchmarkheaven.com/jev-models
* SemIf / OpenJev (open baselines): https://github.com/TheoLeeCJ/SemIf, https://huggingface.co/AlexWortega/openjev
* liteparse: https://github.com/run-llama/liteparse · https://developers.llamaindex.ai/liteparse/

## What's in the notebook

0. Setup · 1. liteparse in 60 seconds · 2. The decision shape · **3. Under the hood** (architecture and training of jev, the
direct-logits decoder, Laya and jeff; what TypeSafe has and hasn't disclosed) · 4–8. one section per task · 9. Fan-out ·
10. Overview · 11. Takeaways.

## Open encoders

* [Laya](https://huggingface.co/convaiinnovations/laya): 421M ModernBERT + decision head, jev-compatible `state`/`questions`
  schema with native fan-out. Runs locally (MPS/CPU). We use the `typed-decisions` checkpoint with the card's
  budget knobs raised (`max_len=2048`, `head_max_len=512`) so full states and 17-option questions fit.
* [jeff](https://github.com/logan-markewich/jeff): a self-hosted server implementing the jev System One API on
  gliformer-large-v1 (400M). Because it speaks the same wire format, the notebook drives it with `JevBackend`
  and just a `base_url`. Run it locally (`JEFF_API_KEYS=devkey uv run jeff` in the jeff repo, then
  `JEFF_BASE_URL=http://localhost:8000`) or on Modal (`deploy/modal_gpu.py`). Set `JEFF_BASE_URL` / `JEFF_API_KEY` in `.env`.
* `GliformerBackend` (naive `classify()` over gliformer-base) is kept in `baselines.py` for reference but is not
  the fair way to use that model; jeff is.

## Notes on the open-model server

* First call after a deploy is a cold start (model load from the `jev-vs-hf-cache` volume, ~2–5 min); the
  container stays warm for 5 min after the last call (`scaledown_window`).
* Qwen3.5 uses gated delta-net layers. The image installs `flash-linear-attention`; `causal_conv1d` still
  falls back to the PyTorch reference kernel (it needs a CUDA build), so expect ~2 s per decision at
  batch 8 on an L4. Swap in vLLM with `prompt_logprobs` if you want production-like latency numbers.
* `DirectLogits(model_name="...")` accepts any HF causal LM (e.g. `Qwen/Qwen3-4B`, `AlexWortega/openjev`
  needs its own NLI head and is not wired up here).
* After editing `modal_app.py`, run `uv run modal app stop -y jev-vs-open` before `modal deploy` if a warm
  container is still up: a live container kept serving the previous code in our testing.

"""Classical, specialised open baselines: no LLM involved.

* language  -> lingua (n-gram statistical detector, 75 languages, pure Python)
* orientation -> tesseract OSD (orientation & script detection) on the page image
"""

from __future__ import annotations

import re
import time
from typing import Sequence

from PIL import Image

from .decisions import Decision, DecisionResult

# ISO 639-1 -> lingua enum name
_LINGUA = {
    "en": "ENGLISH", "de": "GERMAN", "fr": "FRENCH", "es": "SPANISH", "it": "ITALIAN", "pt": "PORTUGUESE",
    "nl": "DUTCH", "pl": "POLISH", "ru": "RUSSIAN", "tr": "TURKISH", "el": "GREEK", "vi": "VIETNAMESE",
    "sv": "SWEDISH", "da": "DANISH", "fi": "FINNISH", "cs": "CZECH", "hu": "HUNGARIAN", "ro": "ROMANIAN",
    "uk": "UKRAINIAN", "id": "INDONESIAN", "ar": "ARABIC", "ja": "JAPANESE", "zh": "CHINESE", "ko": "KOREAN",
    "hi": "HINDI", "th": "THAI", "bg": "BULGARIAN", "sw": "SWAHILI", "ur": "URDU",
}


class LinguaBackend:
    name = "lingua"

    def __init__(self, langs: Sequence[str]):
        from lingua import Language, LanguageDetectorBuilder

        self.langs = list(langs)
        enums = [getattr(Language, _LINGUA[l]) for l in self.langs]
        self.detector = LanguageDetectorBuilder.from_languages(*enums).build()
        self.back = {getattr(Language, _LINGUA[l]): l for l in self.langs}

    def run(self, decisions: Sequence[Decision]) -> list[DecisionResult]:
        out = []
        for d in decisions:
            text = d.state if isinstance(d.state, str) else str(d.state)
            t0 = time.perf_counter()
            conf = self.detector.compute_language_confidence_values(text)
            dt = time.perf_counter() - t0
            probs = {self.back[c.language]: c.value for c in conf if c.language in self.back}
            choice = max(probs, key=probs.get) if probs else "unknown"
            top = sorted(probs.values(), reverse=True) + [0.0, 0.0]
            out.append(DecisionResult(d.id, self.name, choice, probs, top[0] - top[1], dt, None, 0.0, d.truth, dict(d.meta)))
        return out


class TesseractOSDBackend:
    """tesseract --psm 0 on the page image. Needs `osd.traineddata` installed."""

    name = "tesseract-osd"

    def __init__(self, images: dict[str, Image.Image]):
        self.images = images  # decision id -> page image (as scanned, i.e. possibly rotated)

    def run(self, decisions: Sequence[Decision]) -> list[DecisionResult]:
        import pytesseract

        out = []
        for d in decisions:
            img = self.images[d.id]
            t0 = time.perf_counter()
            try:
                osd = pytesseract.image_to_osd(img, config="--psm 0")
                rotate = int(re.search(r"Rotate: (\d+)", osd).group(1))
                conf = float(re.search(r"Orientation confidence: ([\d.]+)", osd).group(1))
            except Exception as e:  # tesseract raises when it finds too little text
                rotate, conf = 0, 0.0
            dt = time.perf_counter() - t0
            # tesseract's "Rotate" is the clockwise correction; our candidates are CCW fixes,
            # so a page scanned theta CW needs candidate theta, and tesseract says Rotate=(360-theta)%360.
            choice = str((360 - rotate) % 360)
            probs = {k: (1.0 if k == choice else 0.0) for k in d.options}
            out.append(DecisionResult(d.id, self.name, choice, probs, min(conf / 10.0, 1.0), dt, None, 0.0, d.truth, dict(d.meta)))
        return out


class HeuristicTriageBackend:
    """What most pipelines actually do: escalate on low OCR confidence or too little text."""

    name = "heuristic"

    def __init__(self, min_confidence: float = 0.85, min_chars: int = 300):
        self.min_confidence, self.min_chars = min_confidence, min_chars

    def run(self, decisions: Sequence[Decision]) -> list[DecisionResult]:
        out = []
        for d in decisions:
            t0 = time.perf_counter()
            ocr = d.state["ocr"]
            upgrade = (ocr["mean_confidence"] or 0) < self.min_confidence or ocr["chars"] < self.min_chars
            choice = "upgrade" if upgrade else "local_ok"
            out.append(DecisionResult(d.id, self.name, choice, {k: float(k == choice) for k in d.options}, 1.0, time.perf_counter() - t0, None, 0.0, d.truth, dict(d.meta)))
        return out


# ------------------------------------------------------------------ encoders (run locally)
class LayaBackend:
    """Laya: ModernBERT-large (421M) + decision head, an open jev-compatible encoder.

    Same state/questions schema as jev, all questions in one forward pass. Runs locally
    (MPS/CPU/CUDA). https://huggingface.co/convaiinnovations/laya
    """

    name = "laya"

    def __init__(self, subfolder: str | None = "typed-decisions", device: str | None = None, max_chars: int | None = None, max_len: int = 2048, head_max_len: int = 512):
        """`subfolder`: None = English checkpoint, "multilingual", or "typed-decisions" (fine-tuned for
        choice/score/noul workflows; best of the three on our tasks). `max_len` / `head_max_len` raise
        Laya's default 512-token budget (the card's documented knobs) so full states and 17-option
        questions fit; `max_chars` optionally trims state strings instead."""
        import laya

        self.agent = laya.load("convaiinnovations/laya", subfolder=subfolder, device=device)
        self.agent.cfg["max_len"] = max_len
        self.agent.cfg["head_max_len"] = head_max_len
        self.device = getattr(self.agent, "device", device)
        self.max_chars = max_chars

    def _shrink(self, state, budget: int | None = None):
        n = budget or self.max_chars
        if isinstance(state, str):
            return {"text": state[:n]}
        if isinstance(state, dict):
            return {k: (v[:n] if isinstance(v, str) else self._shrink(v, n)) for k, v in state.items()}
        if isinstance(state, list):
            return [(v[:n] if isinstance(v, str) else self._shrink(v, n)) for v in state]
        return state

    def _ask(self, state, questions: dict[str, tuple[str, dict, str | None]]):
        qs = {q: {"type": "choice", "instructions": text, "criteria": opts} for q, (text, opts, _) in questions.items()}
        if self.max_chars:
            state = self._shrink(state, self.max_chars // 3 if len(questions) > 1 else None)  # fan-out states are big
        elif isinstance(state, str):
            state = {"text": state}
        t0 = time.perf_counter()
        r = self.agent.predict(state, qs)
        return r, time.perf_counter() - t0

    def run(self, decisions: Sequence[Decision]) -> list[DecisionResult]:
        out = []
        for d in decisions:
            r, dt = self._ask(d.state, {"answer": (d.question, d.options, d.truth)})
            a = r["answers"]["answer"]
            out.append(DecisionResult(d.id, self.name, a["choice"], dict(a["probabilities"]), a.get("confidence"), dt, r.get("usage", {}).get("input_tokens"), 0.0, d.truth, dict(d.meta)))
        return out

    def run_fanout(self, items):
        """All questions of an item in one forward pass (Laya supports the jev fan-out natively)."""
        from .decisions import FanoutStats

        grouped: dict[str, list[DecisionResult]] = {}
        t_all = time.perf_counter()
        toks = 0
        for item in items:
            r, dt = self._ask(item.state, item.questions)
            toks += r.get("usage", {}).get("input_tokens") or 0
            for q, (_, _, truth) in item.questions.items():
                a = r["answers"][q]
                grouped.setdefault(q, []).append(DecisionResult(f"{item.id}#{q}", f"{self.name} (fan-out)", a["choice"], dict(a["probabilities"]), a.get("confidence"), dt, r.get("usage", {}).get("input_tokens"), 0.0, truth, {**item.meta, "question": q}))
        return grouped, FanoutStats(len(items), time.perf_counter() - t_all, toks, 0.0)


class GliformerBackend:
    """gliformer-base-v1: a 264M DeBERTa zero-shot classifier (knowledgator). One call per question.

    Options are passed as "label: description" so it sees the same information as the other
    models; we ask for every label's score (threshold=-1, multi_label) and take the argmax.
    """

    name = "gliformer"

    def __init__(self, model_id: str = "knowledgator/gliformer-base-v1", device: str | None = None):
        import torch
        from gliformer import GLiFormer

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiFormer.from_pretrained(model_id, load_tokenizer=True, map_location=self.device).eval()

    def run(self, decisions: Sequence[Decision]) -> list[DecisionResult]:
        import json

        out = []
        for d in decisions:
            text = d.state if isinstance(d.state, str) else json.dumps(d.state, ensure_ascii=False)
            labels = {f"{k}: {v}" if v else k: k for k, v in d.options.items()}
            t0 = time.perf_counter()
            res = self.model.classify(text, list(labels), threshold=-1.0, multi_label=True)
            dt = time.perf_counter() - t0
            scores = {labels[r["class_name"]]: float(r["score"]) for r in res if r["class_name"] in labels}
            for k in d.options:
                scores.setdefault(k, 0.0)
            total = sum(scores.values()) or 1.0
            probs = {k: v / total for k, v in scores.items()}
            choice = max(probs, key=probs.get)
            top = sorted(probs.values(), reverse=True) + [0.0]
            out.append(DecisionResult(d.id, self.name, choice, probs, top[0] - top[1], dt, None, 0.0, d.truth, dict(d.meta)))
        return out

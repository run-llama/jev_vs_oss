"""One decision spec, several backends.

Every task in the notebook is expressed as a `Decision` (state + question + labelled
options). The same list of decisions is then handed to:

* `JevBackend`          -> TypeSafe's hosted jev model (one Choice question per decision)
* `OpenLogitsBackend`   -> an open model on Modal, read out SemIf-style from next-token logits
* task-specific classical baselines live in `baselines.py`
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

JSONState = str | dict[str, Any] | list[Any]

# https://docs.typesafe.ai/models : $0.042 per million input tokens, output tokens free.
JEV_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
# Modal on-demand GPU pricing (USD/second), used to attribute cost to open-model decisions.
MODAL_GPU_USD_PER_S = {"L4": 0.80 / 3600, "A10G": 1.10 / 3600, "L40S": 1.95 / 3600, "A100-40GB": 2.10 / 3600, "H100": 3.95 / 3600}


@dataclass
class Decision:
    id: str
    state: JSONState
    question: str
    options: dict[str, str | None]  # label -> description (None = undescribed)
    truth: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        """SemIf-style row consumed by the open backend."""
        return {
            "id": self.id,
            "state": self.state,
            "question": self.question,
            "options": [{"id": k, "description": v or k} for k, v in self.options.items()],
        }


@dataclass
class DecisionResult:
    id: str
    backend: str
    choice: str
    probabilities: dict[str, float]
    confidence: float | None
    latency_s: float
    input_tokens: int | None = None
    cost_usd: float | None = None
    truth: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool | None:
        return None if self.truth is None else self.choice == self.truth


def results_to_records(results: Sequence[DecisionResult]) -> list[dict]:
    return [{**asdict(r), "correct": r.correct} for r in results]


# --------------------------------------------------------------------------- jev
class JevBackend:
    """Hosted jev via the typesafe-sdk async client, with bounded concurrency."""

    name = "jev"

    def __init__(self, model: str | None = None, concurrency: int = 16, timeout: float = 60.0, *, base_url: str | None = None, api_key: str | None = None, name: str = "jev", usd_per_input_token: float = JEV_USD_PER_INPUT_TOKEN):
        """Any server speaking the jev wire format: hosted jev by default, or e.g. jeff via `base_url`."""
        self.concurrency = concurrency
        self.model = model
        self.timeout = timeout
        self.base_url = base_url
        self.api_key = api_key
        self.name = name
        self.usd_per_input_token = usd_per_input_token

    def make_client(self):
        """A fresh async client per run: each run gets its own event loop (Jupyter-safe)."""
        from typesafe_sdk import AsyncTypeSafeClient

        return AsyncTypeSafeClient(model=self.model, timeout=self.timeout, base_url=self.base_url, api_key=self.api_key)

    async def _one(self, client, d: Decision, sem: asyncio.Semaphore) -> DecisionResult:
        from typesafe_sdk import Choice

        async with sem:
            t0 = time.perf_counter()
            resp = await client.system_one(
                state=d.state,
                questions={"answer": Choice(instructions=d.question, criteria=d.options)},
            )
            dt = time.perf_counter() - t0
        ans = resp.choices["answer"]
        toks = resp.usage.input_tokens
        return DecisionResult(
            id=d.id,
            backend=self.name,
            choice=ans.choice,
            probabilities=dict(ans.probabilities),
            confidence=ans.confidence,
            latency_s=dt,
            input_tokens=toks,
            cost_usd=(toks or 0) * self.usd_per_input_token,
            truth=d.truth,
            meta={"model": resp.model, **d.meta},
        )

    async def run_async(self, decisions: Sequence[Decision]) -> list[DecisionResult]:
        sem = asyncio.Semaphore(self.concurrency)
        async with self.make_client() as client:
            return list(await asyncio.gather(*(self._one(client, d, sem) for d in decisions)))

    def run(self, decisions: Sequence[Decision]) -> list[DecisionResult]:
        return _run_coro(self.run_async(decisions))


# --------------------------------------------------------------------- open model
class OpenLogitsBackend:
    """Open model served on Modal (see modal_app.py), SemIf-style direct-logits readout.

    Latency is reported per decision as wall time of the remote batch divided by batch
    size, plus the GPU-side forward time recorded remotely; both are kept in `meta`.
    """

    def __init__(self, app_name: str = "jev-vs-open", cls_name: str = "DirectLogits", gpu: str = "L4", batch_size: int = 8, name: str | None = None, model: str | None = None):
        import modal

        self.remote = modal.Cls.from_name(app_name, cls_name)()
        self.gpu = gpu
        self.batch_size = batch_size
        self.model = model
        self.name = name or f"open:{(model or 'Qwen/Qwen3.5-4B').split('/')[-1]}"

    def run(self, decisions: Sequence[Decision]) -> list[DecisionResult]:
        rows = [d.to_row() for d in decisions]
        by_id = {d.id: d for d in decisions}
        out: list[DecisionResult] = []
        chunks = [rows[i : i + self.batch_size] for i in range(0, len(rows), self.batch_size)]
        # Fan chunks out to as many containers as Modal will give us.
        t0 = time.perf_counter()
        # Modal's .map() refuses to run inside a live event loop (Jupyter), so collect it on a worker thread.
        batches = _call_off_loop(lambda: list(self.remote.decide.map(chunks, kwargs={"model": self.model}, order_outputs=True)))
        for batch in batches:
            for r in batch:
                d = by_id[r["id"]]
                probs = dict(zip(r["option_ids"], r["probabilities"]))
                choice = max(probs, key=probs.get)
                out.append(
                    DecisionResult(
                        id=r["id"],
                        backend=self.name,
                        choice=choice,
                        probabilities=probs,
                        confidence=_confidence(probs),
                        latency_s=r["wall_seconds_per_decision"],
                        input_tokens=r["input_tokens"],
                        cost_usd=r["wall_seconds_per_decision"] * MODAL_GPU_USD_PER_S[self.gpu],
                        truth=d.truth,
                        meta={"model": r["model"], "forward_seconds": r["forward_seconds"], **d.meta},
                    )
                )
        self.last_wall_seconds = time.perf_counter() - t0
        return out


def _confidence(probs: dict[str, float]) -> float:
    """Same spirit as jev's confidence: top probability minus the runner-up."""
    vals = sorted(probs.values(), reverse=True)
    return vals[0] - (vals[1] if len(vals) > 1 else 0.0)


def _call_off_loop(fn):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(fn).result()


def _run_coro(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Inside Jupyter: run on a fresh loop in a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def summarize(results: Sequence[DecisionResult]) -> dict:
    lat = sorted(r.latency_s for r in results)
    scored = [r for r in results if r.correct is not None]
    return {
        "n": len(results),
        "accuracy": (sum(r.correct for r in scored) / len(scored)) if scored else None,
        "p50_latency_s": statistics.median(lat) if lat else None,
        "p95_latency_s": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "usd_per_1k": 1000 * sum(r.cost_usd or 0 for r in results) / len(results) if results else None,
        "mean_confidence": statistics.fmean(r.confidence for r in results if r.confidence is not None) if results else None,
    }


def save_results(results: Sequence[DecisionResult], path) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({**asdict(r), "correct": r.correct}) + "\n")


# ----------------------------------------------------------------- fan-out
@dataclass
class FanoutItem:
    """One state, several named questions (jev evaluates them in parallel in one request)."""

    id: str
    state: JSONState
    questions: dict[str, tuple[str, dict[str, str | None], str | None]]  # name -> (question, options, truth)
    meta: dict[str, Any] = field(default_factory=dict)

    def split(self) -> list[Decision]:
        return [Decision(f"{self.id}#{q}", self.state, text, opts, truth, {**self.meta, "question": q}) for q, (text, opts, truth) in self.questions.items()]


@dataclass
class FanoutStats:
    calls: int
    wall_s: float
    input_tokens: int
    cost_usd: float


def fanout_jev(backend: JevBackend, items: Sequence[FanoutItem]) -> tuple[dict[str, list[DecisionResult]], FanoutStats]:
    """All questions of an item in ONE system_one call. Returns results grouped by question name."""
    from typesafe_sdk import Choice

    async def one(client, item: FanoutItem, sem: asyncio.Semaphore):
        async with sem:
            t0 = time.perf_counter()
            resp = await client.system_one(state=item.state, questions={q: Choice(instructions=text, criteria=opts) for q, (text, opts, _) in item.questions.items()})
            dt = time.perf_counter() - t0
        toks = resp.usage.input_tokens or 0
        out = []
        for q, (_, _, truth) in item.questions.items():
            a = resp.choices[q]
            out.append(DecisionResult(f"{item.id}#{q}", f"{backend.name} (fan-out)", a.choice, dict(a.probabilities), a.confidence, dt, toks, toks * backend.usd_per_input_token, truth, {**item.meta, "question": q}))
        return out, dt, toks

    async def run():
        sem = asyncio.Semaphore(backend.concurrency)
        t0 = time.perf_counter()
        async with backend.make_client() as client:
            parts = await asyncio.gather(*(one(client, i, sem) for i in items))
        return parts, time.perf_counter() - t0

    parts, wall = _run_coro(run())
    grouped: dict[str, list[DecisionResult]] = {}
    toks = 0
    for results, _, t in parts:
        toks += t
        for r in results:
            grouped.setdefault(r.meta["question"], []).append(r)
    return grouped, FanoutStats(len(items), wall, toks, toks * backend.usd_per_input_token)


def sequential(backend, items: Sequence[FanoutItem], label: str | None = None) -> tuple[dict[str, list[DecisionResult]], FanoutStats]:
    """Same questions, one decision per call/forward. Works for any backend with .run()."""
    decisions = [d for i in items for d in i.split()]
    t0 = time.perf_counter()
    results = backend.run(decisions)
    wall = time.perf_counter() - t0
    grouped: dict[str, list[DecisionResult]] = {}
    for r in results:
        if label:
            r.backend = label
        grouped.setdefault(r.meta["question"], []).append(r)
    return grouped, FanoutStats(len(decisions), wall, sum(r.input_tokens or 0 for r in results), sum(r.cost_usd or 0 for r in results))

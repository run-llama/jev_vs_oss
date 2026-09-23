"""Scoring tables and charts comparing backends on a task."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .decisions import DecisionResult, results_to_records, summarize

# Categorical slots in palette order (never cycled): jev=blue, open decoder=orange, open encoder=aqua, classical tool=yellow.
COLORS = {"jev": "#2a78d6", "open": "#eb6834", "encoder": "#1baf7a", "classical": "#eda100"}
ENCODERS = ("laya", "jeff", "gliformer", "gliner", "certo")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def color_for(backend: str) -> str:
    if backend.startswith("jev"):
        return COLORS["jev"]
    if backend.startswith("open"):
        return COLORS["open"]
    if backend.split(" ")[0] in ENCODERS:
        return COLORS["encoder"]
    return COLORS["classical"]


def compare(results_by_backend: dict[str, Sequence[DecisionResult]]) -> pd.DataFrame:
    rows = []
    for name, res in results_by_backend.items():
        s = summarize(res)
        rows.append({"backend": name, **s})
    df = pd.DataFrame(rows).set_index("backend")
    return df


def frame(results: Sequence[DecisionResult]) -> pd.DataFrame:
    return pd.DataFrame(results_to_records(results))


def confusion(results: Sequence[DecisionResult]) -> pd.DataFrame:
    df = frame(results)
    return pd.crosstab(df["truth"], df["choice"], rownames=["truth"], colnames=["predicted"])


def errors(results: Sequence[DecisionResult], n: int = 10) -> pd.DataFrame:
    df = frame(results)
    df = df[df["correct"] == False]  # noqa: E712
    return df[["id", "truth", "choice", "confidence"]].head(n)


def _style(ax, title: str):
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_task(df: pd.DataFrame, task: str):
    """Three small bar charts: accuracy, p50 latency, $/1k decisions. One axis each."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    fig.patch.set_facecolor("#fcfcfb")
    specs = [("accuracy", "Accuracy", "{:.0%}"), ("p50_latency_s", "p50 latency (s / decision)", "{:.2f}s"), ("usd_per_1k", "Cost (USD / 1k decisions)", "${:.3f}")]
    for ax, (col, title, fmt) in zip(axes, specs):
        ax.set_facecolor("#fcfcfb")
        vals = df[col].fillna(0)
        bars = ax.bar(df.index, vals, color=[color_for(b) for b in df.index], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
        _style(ax, title)
        if col == "accuracy":
            ax.set_ylim(0, 1.08)
    fig.suptitle(task, x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_overview(per_task: dict[str, pd.DataFrame]):
    """Accuracy per task, grouped by backend family."""
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = list(per_task)
    backends = sorted({b for df in per_task.values() for b in df.index}, key=lambda b: (not b.startswith("jev"), not b.startswith("open"), b.split(" ")[0] not in ENCODERS, b))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = np.arange(len(tasks))
    w = 0.8 / len(backends)
    seen: set[str] = set()
    for i, b in enumerate(backends):
        vals = [per_task[t].loc[b, "accuracy"] if b in per_task[t].index else np.nan for t in tasks]
        family = "jev" if b.startswith("jev") else "open decoder" if b.startswith("open") else "open encoder" if b.split(" ")[0] in ENCODERS else "specialised tool"
        bars = ax.bar(x + (i - (len(backends) - 1) / 2) * w, vals, width=w * 0.92, color=color_for(b), label=family if family not in seen else None)
        seen.add(family)
        for bar, v in zip(bars, vals):
            if v == v:
                ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, tasks)
    ax.set_ylim(0, 1.1)
    _style(ax, "Accuracy by task and backend")
    ax.legend(frameon=False, fontsize=9, ncol=len(backends), loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    return fig

"""Plots for the report (specs.md §10): the three required charts plus two diagnostics.

Every function takes plain result rows (dicts loaded from the results JSONL
files) and writes one PNG. It also writes the CSV of the exact points drawn
beside it, so every value can be read without relying on colour.

Styling follows a small fixed palette, applied in a fixed order and validated
for colour-vision deficiency. Lines are 2px with direct end labels, axes and
grids are recessive, and every chart has one y-axis.

Depends on: matplotlib (Agg backend, no display needed) and the standard library.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _figure(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: Path, rows: list[dict]) -> None:
    """Write the PNG and a CSV of the plotted points next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    if rows:
        with path.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _end_labels(ax, ends: list[tuple[float, float, str]], min_gap_px: float = 11) -> None:
    """Label line ends in ink, skipping any label that would collide with one already placed.

    Call this after the axis scales are set. The legend still names every line,
    so a skipped label loses no information.
    """
    ax.figure.canvas.draw()
    placed: list[float] = []
    for x, y, text in sorted(ends, key=lambda e: ax.transData.transform((e[0], e[1]))[1]):
        y_px = ax.transData.transform((x, y))[1]
        if all(abs(y_px - other) >= min_gap_px for other in placed):
            ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_SECONDARY)
            placed.append(y_px)


def _k_label(k) -> str:
    return "k = optimal" if k in (None, "opt") else f"k = {k}"


def _by_k(index_rows: list[dict]) -> dict:
    """Group grid index rows by k, summing memory and averaging FPR over documents, per R."""
    groups: dict = defaultdict(lambda: defaultdict(list))
    for row in index_rows:
        if row["index"] == "rowmajor":
            groups[row["k_label"]][row["R"]].append(row)
    return groups


def plot_memory_vs_fpr(index_rows: list[dict], path: Path) -> None:
    """Required plot (a): Bloom memory vs FPR, measured (solid) and theoretical (dashed), one line per k."""
    fig, ax = _figure("Bloom memory vs false-positive rate",
                      "Bloom filter memory, all 4 documents (KiB, log scale)",
                      "False-positive rate (log scale)")
    points, ends = [], []
    for i, (k, by_R) in enumerate(sorted(_by_k(index_rows).items(), key=lambda kv: str(kv[0]))):
        color = SERIES[i % len(SERIES)]
        xs, measured, theory = [], [], []
        for R in sorted(by_R):
            rows = by_R[R]
            memory_kib = sum(r["bloom_bits"] for r in rows) / 8 / 1024
            m = sum(r["fpr_measured"] for r in rows) / len(rows)
            t = sum(r["fpr_theoretical"] for r in rows) / len(rows)
            xs.append(memory_kib)
            measured.append(m)
            theory.append(t)
            points.append({"k": k, "R": R, "memory_kib": memory_kib, "fpr_measured": m, "fpr_theoretical": t})
        ax.plot(xs, theory, "--", color=color, linewidth=1.2, alpha=0.8)
        ax.plot(xs, measured, "-o", color=color, linewidth=2, markersize=5, label=_k_label(k))
        ends.append((xs[-1], measured[-1], _k_label(k)))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _end_labels(ax, ends)
    ax.plot([], [], "--", color=MUTED, label="theoretical (dashed)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _save(fig, path, points)


def plot_memory_vs_tokens(index_rows: list[dict], grid_rows: list[dict], path: Path,
                          baseline_tokens: float | None, estimated: bool = True) -> None:
    """Required plot (b): Bloom memory vs mean LLM input tokens per question, one line per k.

    In offline runs the token counts are the ``ceil(chars/4)`` estimate, and the y-axis says so.
    """
    tokens: dict = defaultdict(list)
    for row in grid_rows:
        if row["index"] == "rowmajor":
            tokens[(row["k_label"], row["R"])].append(row["prompt_tokens"])
    memory: dict = defaultdict(int)
    for row in index_rows:
        if row["index"] == "rowmajor":
            memory[(row["k_label"], row["R"])] += row["bloom_bits"]

    label = "estimated " if estimated else ""
    fig, ax = _figure("Bloom memory vs LLM input tokens",
                      "Bloom filter memory, all 4 documents (KiB, log scale)",
                      f"Mean {label}input tokens per question")
    points, ends = [], []
    ks = sorted({k for k, _ in tokens}, key=str)
    for i, k in enumerate(ks):
        color = SERIES[i % len(SERIES)]
        Rs = sorted(R for kk, R in tokens if kk == k)
        xs = [memory[(k, R)] / 8 / 1024 for R in Rs]
        ys = [sum(tokens[(k, R)]) / len(tokens[(k, R)]) for R in Rs]
        points += [{"k": k, "R": R, "memory_kib": x, "mean_input_tokens": y} for R, x, y in zip(Rs, xs, ys)]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=5, label=_k_label(k))
        ends.append((xs[-1], ys[-1], _k_label(k)))
    ax.set_xscale("log", base=2)
    _end_labels(ax, ends)
    if baseline_tokens is not None:
        ax.set_title(f"Bloom memory vs LLM input tokens (full context ≈ {baseline_tokens:,.0f} tokens per question)",
                     color=INK, loc="left", fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _save(fig, path, points)


def plot_tokens_vs_accuracy(summary_rows: list[dict], path: Path, baseline_config: str = "full") -> None:
    """Required plot (c): mean input tokens vs strict accuracy per config, with the baseline marked.

    Configs are grouped into at most three coloured families, since a scatter
    can only keep three colours apart for every reader.
    """
    fig, ax = _figure("LLM input tokens vs answer accuracy",
                      "Mean input tokens per question (log scale)", "Strict accuracy")
    families = {"R sweep": [], "exact set": [], "ablations & budgets": []}
    baseline = None
    for row in summary_rows:
        if row["group_by"] != "all":
            continue
        if row["config"] == baseline_config:
            baseline = row
        elif row["config"] == "exact":
            families["exact set"].append(row)
        elif row["config"] == "core" or row["config"].startswith("R"):
            families["R sweep"].append(row)
        else:
            families["ablations & budgets"].append(row)
    for i, (family, rows) in enumerate(families.items()):
        if not rows:
            continue
        xs = [r["prompt_tokens"] for r in rows]
        ys = [r["correct_strict"] for r in rows]
        ax.plot(xs, ys, "o", color=SERIES[i], markersize=7, label=family,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for r in rows:
            ax.annotate(r["config"], (r["prompt_tokens"], r["correct_strict"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color=INK_SECONDARY)
    if baseline:
        ax.plot([baseline["prompt_tokens"]], [baseline["correct_strict"]], "*", color=INK,
                markersize=12, label="full context (baseline)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower right")
    rows = [{"config": r["config"], "mean_input_tokens": r["prompt_tokens"],
             "accuracy": r["correct_strict"]} for r in summary_rows if r["group_by"] == "all"]
    _save(fig, path, rows)


def plot_recall_vs_fpr(index_rows: list[dict], grid_rows: list[dict], path: Path) -> None:
    """Diagnostic: how membership FPR erodes retrieval, as gold-span recall vs measured FPR per k."""
    recall: dict = defaultdict(list)
    for row in grid_rows:
        if row["index"] == "rowmajor":
            recall[(row["k_label"], row["R"])].append(row["gold_span_recall"])
    fpr: dict = defaultdict(list)
    for row in index_rows:
        if row["index"] == "rowmajor":
            fpr[(row["k_label"], row["R"])].append(row["fpr_measured"])
    fig, ax = _figure("Retrieval quality vs false-positive rate",
                      "Measured false-positive rate (log scale)", "Mean gold-span recall")
    points, ends = [], []
    for i, k in enumerate(sorted({k for k, _ in recall}, key=str)):
        color = SERIES[i % len(SERIES)]
        Rs = sorted(R for kk, R in recall if kk == k)
        pts = sorted(
            (sum(fpr[(k, R)]) / len(fpr[(k, R)]), sum(recall[(k, R)]) / len(recall[(k, R)]), R)
            for R in Rs
        )
        pts = [p for p in pts if p[0] > 0]  # a log axis cannot show FPR = 0
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        points += [{"k": k, "R": p[2], "fpr_measured": p[0], "gold_span_recall": p[1]} for p in pts]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=5, label=_k_label(k))
        ends.append((xs[-1], ys[-1], _k_label(k)))
    ax.set_xscale("log")
    _end_labels(ax, ends)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    _save(fig, path, points)


def plot_config_bars(summary_rows: list[dict], path: Path, metrics: tuple[str, str], titles: tuple[str, str],
                     order: list[str] | None = None) -> None:
    """Diagnostic: per-config horizontal bars for two metrics, in side-by-side panels (never a dual axis).

    Bars run top-down in ``order``. Configs missing from ``order`` follow alphabetically.
    """
    order = order or []
    rank = {name: i for i, name in enumerate(order)}
    rows = sorted((r for r in summary_rows if r["group_by"] == "all"),
                  key=lambda r: (rank.get(r["config"], len(order)), r["config"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.4 * len(rows) + 1.5), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    names = [r["config"] for r in rows]
    for ax, metric, title in zip(axes, metrics, titles):
        ax.set_facecolor(SURFACE)
        values = [r[metric] for r in rows]
        ax.barh(names, values, color=SERIES[0], height=0.6, edgecolor=SURFACE, linewidth=2)
        for y, v in enumerate(values):
            ax.annotate(f"{v:,.2f}" if v < 10 else f"{v:,.0f}", (v, y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color=INK_SECONDARY)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()  # first config at the top (the y-axis is shared)
    _save(fig, path, [{"config": r["config"], metrics[0]: r[metrics[0]], metrics[1]: r[metrics[1]]} for r in rows])

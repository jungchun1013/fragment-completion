#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cav_plot.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Visualize CAV training results: layer × concept test-accuracy heatmaps.

Reads the .pt file produced by ``cav_train.py`` and saves a 1×3 heatmap
panel (color / material / function) showing per-layer test accuracy for
every concept bin. Bins within each panel are sorted by argmax layer so
the early-vs-late gradient is visually obvious.

Usage:
    uv run python -m experiments.exp3.cav_plot
    uv run python -m experiments.exp3.cav_plot --cavs data/.../cavs/clip_L14.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import PLOT_STYLE as PS


DIM_ORDER = ("color", "material", "function")


def _stack_dim(
    cavs: dict, dim: str,
) -> tuple[list[str], np.ndarray]:
    """Collect [bins × layers] test_acc matrix for one dimension.

    Bins are sorted by argmax layer (early peaks first), so the heatmap
    reads top-to-bottom as early → late layer reliance.
    """
    rows: list[tuple[str, np.ndarray]] = []
    for (d, bin_name), v in cavs.items():
        if d != dim:
            continue
        rows.append((bin_name, v["test_acc"].numpy()))
    if not rows:
        return [], np.zeros((0, 0))
    rows.sort(key=lambda r: int(np.argmax(r[1])))
    names = [r[0] for r in rows]
    matrix = np.stack([r[1] for r in rows], axis=0)
    return names, matrix


def plot_layer_concept_heatmap(
    payload: dict, save_path: Path,
) -> None:
    """Save a 1×3 heatmap (color / material / function) of per-layer test_acc."""
    cavs = payload["cavs"]
    num_layers = payload["num_layers"]
    model_name = payload.get("model", "?")

    by_dim = {dim: _stack_dim(cavs, dim) for dim in DIM_ORDER}

    fig_w = PS["subplot_size"][0] * len(DIM_ORDER) + 1.0
    fig_h = 7.0

    fig, axes = plt.subplots(1, len(DIM_ORDER), figsize=(fig_w, fig_h))

    vmin, vmax = 0.5, 1.0  # chance level is 0.5 for balanced binary CAV
    cmap = "Blues"

    im = None
    for ax, dim in zip(axes, DIM_ORDER):
        names, matrix = by_dim[dim]
        if len(names) == 0:
            ax.set_title(f"{dim} (no bins)", fontsize=PS["subplot_title_fontsize"])
            ax.axis("off")
            continue

        im = ax.imshow(
            matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=PS["tick_labelsize"] - 4)
        ax.set_xlabel("Layer", fontsize=PS["label_fontsize"])
        ax.set_title(dim, fontsize=PS["subplot_title_fontsize"])
        ax.tick_params(axis="x", labelsize=PS["tick_labelsize"] - 2)

        for i in range(matrix.shape[0]):
            best = int(np.argmax(matrix[i]))
            ax.add_patch(plt.Rectangle(
                (best - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=1.5,
            ))

    fig.subplots_adjust(left=0.06, right=0.92, top=0.90, bottom=0.10, wspace=0.35)
    if im is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Test accuracy", fontsize=PS["label_fontsize"])
        cbar.ax.tick_params(labelsize=PS["tick_labelsize"] - 2)

    fig.suptitle(
        f"CAV per-layer test accuracy — {model_name}",
        fontsize=PS["suptitle_fontsize"],
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"])
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_dim_mean_curves(
    payload: dict, save_path: Path,
) -> None:
    """Per-dim mean test_acc curve over layers (with std band).

    This is the headline figure for the early-color / late-semantic
    hypothesis: each dimension's mean curve peaks at a different layer.
    """
    cavs = payload["cavs"]
    num_layers = payload["num_layers"]
    model_name = payload.get("model", "?")

    fig, ax = plt.subplots(figsize=PS["subplot_size"])

    colors = plt.cm.tab10.colors
    for idx, dim in enumerate(DIM_ORDER):
        _, matrix = _stack_dim(cavs, dim)
        if matrix.size == 0:
            continue
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        layers = np.arange(num_layers)
        ax.plot(
            layers, mean, marker=PS["marker"], markersize=PS["markersize"],
            linewidth=PS["linewidth"], color=colors[idx], label=dim,
        )
        ax.fill_between(
            layers, mean - std, mean + std,
            alpha=PS["std_alpha"], color=colors[idx],
        )

    ax.set_xlabel("Layer", fontsize=PS["label_fontsize"])
    ax.set_ylabel("CAV test accuracy", fontsize=PS["label_fontsize"])
    ax.set_title(
        f"Concept-dim reliance across layers — {model_name}",
        fontsize=PS["subplot_title_fontsize"],
    )
    ax.tick_params(labelsize=PS["tick_labelsize"])
    ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.30),
        ncol=len(DIM_ORDER), fontsize=PS["legend_fontsize"],
        frameon=False,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_ablation_double_diff(
    ablation_path: Path, save_path: Path,
) -> None:
    """Bar chart of double-difference (pos_drop - neg_drop) per concept.

    A positive bar means: ablating this CAV hurts category classification
    on instances that HAVE this concept more than it hurts instances that
    don't — i.e., the model relies on this direction to recognize those
    instances. Bars are grouped by dim (color/material/function) and sorted
    descending within each group.
    """
    with open(ablation_path) as f:
        data = json.load(f)

    model_name = data.get("model", "?")
    baseline_acc = data.get("baseline_overall_acc", 0)

    # Group by dim
    by_dim: dict[str, list[tuple[str, float, float, float, int]]] = {
        d: [] for d in DIM_ORDER
    }
    for key, v in data["results"].items():
        dim, bin_name = key.split("|", 1)
        if dim not in by_dim:
            continue
        by_dim[dim].append((
            bin_name, v["double_diff"], v["pos_drop"], v["neg_drop"],
            v["best_layer"],
        ))
    for dim in by_dim:
        by_dim[dim].sort(key=lambda r: -r[1])

    fig_w = PS["subplot_size"][0] * len(DIM_ORDER) + 1.0
    fig_h = 6.0
    fig, axes = plt.subplots(1, len(DIM_ORDER), figsize=(fig_w, fig_h))

    colors = plt.cm.tab10.colors
    for idx, (ax, dim) in enumerate(zip(axes, DIM_ORDER)):
        rows = by_dim[dim]
        if not rows:
            ax.set_title(f"{dim} (empty)")
            ax.axis("off")
            continue

        names = [r[0] for r in rows]
        diffs = [r[1] for r in rows]
        layers = [r[4] for r in rows]
        y = np.arange(len(names))
        bars = ax.barh(y, diffs, color=colors[idx], alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{n}  (L{l})" for n, l in zip(names, layers)],
            fontsize=PS["tick_labelsize"] - 4,
        )
        ax.invert_yaxis()
        ax.set_xlabel(r"$\Delta$ acc (pos_drop $-$ neg_drop)",
                      fontsize=PS["label_fontsize"] - 2)
        ax.set_title(dim, fontsize=PS["subplot_title_fontsize"])
        ax.tick_params(axis="x", labelsize=PS["tick_labelsize"] - 2)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Concept-specific ablation impact — {model_name} "
        f"(baseline acc {baseline_acc:.3f})",
        fontsize=PS["suptitle_fontsize"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"])
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_ablation_pos_neg_scatter(
    ablation_path: Path, save_path: Path,
) -> None:
    """Scatter: pos_drop vs neg_drop, colored by dim.

    Points above the y=x line are concept-specific (hurt positives more).
    A diagonal cluster around (0, 0) means the CAV doesn't actually drive
    classification — useful for spotting noise CAVs.
    """
    with open(ablation_path) as f:
        data = json.load(f)
    model_name = data.get("model", "?")

    fig, ax = plt.subplots(figsize=PS["subplot_size"])
    colors = plt.cm.tab10.colors
    dim_to_color = {d: colors[i] for i, d in enumerate(DIM_ORDER)}

    for key, v in data["results"].items():
        dim, bin_name = key.split("|", 1)
        ax.scatter(
            v["neg_drop"], v["pos_drop"],
            color=dim_to_color.get(dim, "gray"),
            s=60, alpha=0.8, edgecolor="white", linewidth=0.5,
        )
        ax.annotate(
            bin_name, (v["neg_drop"], v["pos_drop"]),
            fontsize=8, alpha=0.7,
            xytext=(3, 3), textcoords="offset points",
        )

    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("neg_drop (other-concept instances)",
                  fontsize=PS["label_fontsize"])
    ax.set_ylabel("pos_drop (this-concept instances)",
                  fontsize=PS["label_fontsize"])
    ax.set_title(f"CAV ablation impact — {model_name}",
                 fontsize=PS["subplot_title_fontsize"])

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=dim_to_color[d], markersize=10, label=d)
        for d in DIM_ORDER
    ]
    ax.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=len(DIM_ORDER), fontsize=PS["legend_fontsize"], frameon=False,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_degradation_curves(
    degrade_path: Path, save_path: Path,
) -> None:
    """1×3 panels (color/material/function) of double_diff vs visibility.

    Each panel shows one curve per CAV in that dim. The hypothesis is that
    semantic CAVs (function, late-layer material) should grow toward low
    visibility, while low-level CAVs (color) stay flat. Curves are colored
    by per-dim tab10 ramp; the line for each CAV is constant alpha.
    """
    with open(degrade_path) as f:
        data = json.load(f)
    model_name = data.get("model", "?")

    by_dim: dict[str, list[tuple[str, list[float], list[float]]]] = {
        d: [] for d in DIM_ORDER
    }
    for key, levels in data["results"].items():
        if not levels:
            continue
        dim, bin_name = key.split("|", 1)
        if dim not in by_dim:
            continue
        levels_sorted = sorted(levels, key=lambda r: r["vis"])
        vis = [r["vis"] for r in levels_sorted]
        diff = [r["double_diff"] for r in levels_sorted]
        by_dim[dim].append((bin_name, vis, diff))

    fig_w = PS["subplot_size"][0] * len(DIM_ORDER) + 1.0
    fig_h = 5.5
    fig, axes = plt.subplots(
        1, len(DIM_ORDER), figsize=(fig_w, fig_h), sharey=True,
    )

    for ax, dim in zip(axes, DIM_ORDER):
        rows = by_dim[dim]
        if not rows:
            ax.set_title(f"{dim} (empty)")
            ax.axis("off")
            continue
        n = len(rows)
        cmap = plt.cm.get_cmap("tab20", max(n, 1))
        for i, (bin_name, vis, diff) in enumerate(rows):
            ax.plot(
                vis, diff,
                marker=PS["marker"], markersize=PS["markersize"] - 1,
                linewidth=PS["linewidth"],
                color=cmap(i), label=bin_name,
            )
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax.set_xlabel("Visibility ratio", fontsize=PS["label_fontsize"])
        ax.set_title(dim, fontsize=PS["subplot_title_fontsize"])
        ax.tick_params(labelsize=PS["tick_labelsize"] - 2)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, -0.55),
            ncol=2, fontsize=PS["legend_fontsize"] - 4, frameon=False,
        )

    axes[0].set_ylabel(r"$\Delta$ (pos_drop $-$ neg_drop)",
                       fontsize=PS["label_fontsize"])
    fig.suptitle(
        f"Concept ablation impact across visibility — {model_name}",
        fontsize=PS["suptitle_fontsize"],
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"])
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_degradation_dim_means(
    degrade_path: Path, save_path: Path,
) -> None:
    """Per-dim mean double_diff curve across visibility (headline figure)."""
    with open(degrade_path) as f:
        data = json.load(f)
    model_name = data.get("model", "?")

    fig, ax = plt.subplots(figsize=PS["subplot_size"])
    colors = plt.cm.tab10.colors

    for idx, dim in enumerate(DIM_ORDER):
        rows = []
        for key, levels in data["results"].items():
            if not levels or key.split("|", 1)[0] != dim:
                continue
            ls = sorted(levels, key=lambda r: r["vis"])
            rows.append([r["double_diff"] for r in ls])
        if not rows:
            continue
        arr = np.array(rows)  # [n_cavs, n_levels]
        vis = sorted({r["vis"] for k, ll in data["results"].items()
                      for r in ll if k.split("|", 1)[0] == dim})
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(
            vis, mean, marker=PS["marker"], markersize=PS["markersize"],
            linewidth=PS["linewidth"], color=colors[idx], label=dim,
        )
        ax.fill_between(
            vis, mean - std, mean + std,
            alpha=PS["std_alpha"], color=colors[idx],
        )

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Visibility ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel(r"Mean $\Delta$ (pos_drop $-$ neg_drop)",
                  fontsize=PS["label_fontsize"])
    ax.set_title(
        f"Concept reliance under degradation — {model_name}",
        fontsize=PS["subplot_title_fontsize"],
    )
    ax.tick_params(labelsize=PS["tick_labelsize"])
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.30),
        ncol=len(DIM_ORDER), fontsize=PS["legend_fontsize"], frameon=False,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_subspace_ablation(
    subspace_path: Path, save_path: Path,
) -> None:
    """Plot subspace ablation: per-condition acc curves AND net drop curves.

    Two-panel figure:
      Left:  acc vs vis for baseline + each ablation condition.
      Right: concept-specific net drop = ablate(concept) - ablate(random)
             vs vis, one curve per dim.
    """
    with open(subspace_path) as f:
        data = json.load(f)
    model_name = data.get("model", "?")
    ablate_layer = data.get("ablate_layer", -1)

    levels = sorted(data["levels"], key=lambda r: r["vis"])
    vis = [r["vis"] for r in levels]
    levs = [r["level"] for r in levels]

    baseline = [data["baseline_acc"][str(L)] if str(L) in data["baseline_acc"]
                else data["baseline_acc"][L] for L in levs]

    def col(cond):
        d = data["ablated_acc"][cond]
        return [d[str(L)] if str(L) in d else d[L] for L in levs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Left: raw acc curves ----
    ax = axes[0]
    ax.plot(vis, baseline, color="black", linewidth=2.5,
            marker="o", label="baseline")

    style = {
        "color":     ("color",     plt.cm.tab10.colors[0], "-"),
        "material":  ("material",  plt.cm.tab10.colors[1], "-"),
        "function":  ("function",  plt.cm.tab10.colors[2], "-"),
        "all":       ("all (k=Σ)", "darkred",              "-"),
    }
    for cond, (label, color, ls) in style.items():
        if cond in data["ablated_acc"]:
            ax.plot(vis, col(cond), color=color, linewidth=2,
                    marker="o", linestyle=ls, label=f"ablate {label}")
        rcond = f"random_{cond}"
        if rcond in data["ablated_acc"]:
            ax.plot(vis, col(rcond), color=color, linewidth=1.5,
                    marker="x", linestyle="--", alpha=0.6,
                    label=f"random k={data['subspace_ranks'][cond]}")

    ax.set_xlabel("Visibility ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel("Categorization acc", fontsize=PS["label_fontsize"])
    ax.set_title(f"Subspace ablation @ L{ablate_layer}",
                 fontsize=PS["subplot_title_fontsize"])
    ax.tick_params(labelsize=PS["tick_labelsize"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=PS["legend_fontsize"] - 4, loc="upper left", ncol=2)

    # ---- Right: net (concept-specific) drop curves ----
    ax = axes[1]
    for cond, (label, color, _) in style.items():
        rcond = f"random_{cond}"
        if cond not in data["ablated_acc"] or rcond not in data["ablated_acc"]:
            continue
        net = [r - c for r, c in zip(col(rcond), col(cond))]
        ax.plot(vis, net, color=color, linewidth=PS["linewidth"] + 0.5,
                marker=PS["marker"], markersize=PS["markersize"],
                label=f"{label}")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Visibility ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel("Concept-specific drop\n(random − concept)",
                  fontsize=PS["label_fontsize"] - 2)
    ax.set_title("Net contribution per dim",
                 fontsize=PS["subplot_title_fontsize"])
    ax.tick_params(labelsize=PS["tick_labelsize"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=PS["legend_fontsize"] - 2, loc="best")

    fig.suptitle(
        f"Concept-subspace ablation across visibility — {model_name}",
        fontsize=PS["suptitle_fontsize"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"])
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_subspace_retrieval(
    retrieval_path: Path, save_path: Path,
) -> None:
    """Three-panel plot for image vs text retrieval ablation comparison.

    Panels:
      Left:   net drop on IMAGE retrieval, one curve per dim.
      Middle: net drop on TEXT retrieval, one curve per dim.
      Right:  difference (text_net − image_net), positive = text hurt more.
    """
    with open(retrieval_path) as f:
        data = json.load(f)
    model_name = data.get("model", "?")
    ablate_layer = data.get("ablate_layer", -1)

    levels = sorted(data["levels"], key=lambda r: r["vis"])
    vis = [r["vis"] for r in levels]
    levs = [r["level"] for r in levels]

    def get_metric(cond: str, metric: str) -> list[float]:
        d = data["ablated"][cond][metric]
        return [d[str(L)] if str(L) in d else d[L] for L in levs]

    base_img = [data["baseline"]["image_r1"][str(L)] for L in levs]
    base_txt = [data["baseline"]["text_r1"][str(L)] for L in levs]

    style = {
        "color":    ("color",     plt.cm.tab10.colors[0]),
        "material": ("material",  plt.cm.tab10.colors[1]),
        "function": ("function",  plt.cm.tab10.colors[2]),
        "all":      ("all (k=Σ)", "darkred"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def net_drop(cond: str, metric: str) -> list[float]:
        a = get_metric(cond, metric)
        r = get_metric(f"random_{cond}", metric)
        return [ri - ai for ri, ai in zip(r, a)]

    # Panel 1: image retrieval net drop
    ax = axes[0]
    for cond, (label, color) in style.items():
        ax.plot(vis, net_drop(cond, "image_r1"),
                marker=PS["marker"], markersize=PS["markersize"],
                linewidth=PS["linewidth"], color=color, label=label)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Visibility ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel("Net drop (random − concept)",
                  fontsize=PS["label_fontsize"] - 2)
    ax.set_title("Image retrieval", fontsize=PS["subplot_title_fontsize"])
    ax.tick_params(labelsize=PS["tick_labelsize"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=PS["legend_fontsize"] - 2, loc="best")

    # Panel 2: text retrieval net drop
    ax = axes[1]
    for cond, (label, color) in style.items():
        ax.plot(vis, net_drop(cond, "text_r1"),
                marker=PS["marker"], markersize=PS["markersize"],
                linewidth=PS["linewidth"], color=color, label=label)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Visibility ratio", fontsize=PS["label_fontsize"])
    ax.set_title("Text retrieval", fontsize=PS["subplot_title_fontsize"])
    ax.tick_params(labelsize=PS["tick_labelsize"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=PS["legend_fontsize"] - 2, loc="best")

    # Panel 3: text - image
    ax = axes[2]
    for cond, (label, color) in style.items():
        diff = [t - i for t, i in zip(
            net_drop(cond, "text_r1"), net_drop(cond, "image_r1"),
        )]
        ax.plot(vis, diff,
                marker=PS["marker"], markersize=PS["markersize"],
                linewidth=PS["linewidth"], color=color, label=label)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Visibility ratio", fontsize=PS["label_fontsize"])
    ax.set_ylabel("text_net − image_net",
                  fontsize=PS["label_fontsize"] - 2)
    ax.set_title("Text hurt more ↑    Image hurt more ↓",
                 fontsize=PS["subplot_title_fontsize"] - 2)
    ax.tick_params(labelsize=PS["tick_labelsize"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=PS["legend_fontsize"] - 2, loc="best")

    # Share y-limits between left two panels for fair comparison
    y_lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_hi = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(y_lo, y_hi)
    axes[1].set_ylim(y_lo, y_hi)

    fig.suptitle(
        f"Subspace ablation @ L{ablate_layer} — image vs text retrieval — "
        f"{model_name}",
        fontsize=PS["suptitle_fontsize"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"])
    plt.close(fig)
    print(f"Saved {save_path}")


def _patching_recovery_per_group(
    payload: dict, level: int, group_image_ids: list[str],
) -> tuple[list[int], list[float]]:
    """Compute (L_patch list, recovery curve) for one image group at one level.

    recovery(L_patch) = mean_g(correct_patched - correct_baseline)
                      / max(eps, mean_g(correct_clean - correct_baseline))
    """
    image_ids = payload["image_ids"]
    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    idx = np.array([id_to_idx[iid] for iid in group_image_ids if iid in id_to_idx])
    if len(idx) == 0:
        return list(range(payload["num_layers"])), [0.0] * payload["num_layers"]

    correct_clean = np.array(payload["correct_clean"])[idx]
    correct_baseline = np.array(
        payload["correct_baseline_per_level"][level]
        if level in payload["correct_baseline_per_level"]
        else payload["correct_baseline_per_level"][str(level)]
    )[idx]
    patched_lookup = (
        payload["correct_patched_per_level"][level]
        if level in payload["correct_patched_per_level"]
        else payload["correct_patched_per_level"][str(level)]
    )
    correct_patched = np.array(patched_lookup)[:, idx]  # [L_patch, n_g]

    delta_clean = correct_clean.mean() - correct_baseline.mean()
    eps = 1e-6
    layers = list(range(payload["num_layers"]))
    recovery = []
    for L_patch in layers:
        delta_patch = correct_patched[L_patch].mean() - correct_baseline.mean()
        recovery.append(float(delta_patch / max(eps, delta_clean)))
    return layers, recovery


def _group_image_ids_by_dim(
    payload: dict, dim: str,
) -> dict[str, list[str]]:
    """Return {bin_name: [image_ids]} for one concept dim, only labeled imgs."""
    bins = payload["concept_bins"]
    grouped: dict[str, list[str]] = {}
    for iid, dims in bins.items():
        b = dims.get(dim)
        if b is None:
            continue
        grouped.setdefault(b, []).append(iid)
    return grouped


def plot_patching_recovery_curves(
    payload: dict, save_path: Path, level_to_show: int = 4,
) -> None:
    """1×3 panels (color/material/function) — recovery vs L_patch.

    One curve per concept bin within each dim, plus a thick black "all
    images" reference line. Single representative L_mask (default L=4,
    mid-corruption). The headline figure for "where does each concept
    enter the residual stream" — bins should reach high recovery at
    progressively later L_patch as we move from color → material → function.
    """
    model_name = payload.get("model", "?")
    num_layers = payload["num_layers"]
    vis = payload["vis_per_level"].get(level_to_show) or \
          payload["vis_per_level"].get(str(level_to_show))

    fig_w = PS["subplot_size"][0] * len(DIM_ORDER) + 1.0
    fig, axes = plt.subplots(1, len(DIM_ORDER), figsize=(fig_w, 5.5),
                             sharey=True)

    all_image_ids = payload["image_ids"]
    layers_all, recov_all = _patching_recovery_per_group(
        payload, level_to_show, all_image_ids,
    )

    for ax, dim in zip(axes, DIM_ORDER):
        groups = _group_image_ids_by_dim(payload, dim)
        if not groups:
            ax.set_title(f"{dim} (no labels)")
            ax.axis("off")
            continue
        # Sort bins by group size for consistent color use
        sorted_bins = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        n_bins = len(sorted_bins)
        cmap = plt.get_cmap("tab20", max(n_bins, 1))

        for i, (bin_name, ids) in enumerate(sorted_bins):
            if len(ids) < 5:
                continue  # too few to be meaningful
            layers, recov = _patching_recovery_per_group(
                payload, level_to_show, ids,
            )
            ax.plot(
                layers, recov,
                marker=PS["marker"], markersize=PS["markersize"] - 2,
                linewidth=PS["linewidth"] - 0.5, color=cmap(i),
                label=f"{bin_name} (n={len(ids)})", alpha=0.85,
            )

        # Overall reference line
        ax.plot(
            layers_all, recov_all,
            color="black", linewidth=PS["linewidth"] + 1.0,
            marker=PS["marker"], markersize=PS["markersize"],
            label="all", zorder=10,
        )

        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.4)
        ax.axhline(1, color="gray", linewidth=0.6, alpha=0.4, linestyle=":")
        ax.set_xlabel("L_patch", fontsize=PS["label_fontsize"])
        ax.set_title(dim, fontsize=PS["subplot_title_fontsize"])
        ax.tick_params(labelsize=PS["tick_labelsize"])
        ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
        ax.set_xlim(-0.5, num_layers - 0.5)
        ax.set_ylim(-0.2, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, -0.55),
            ncol=2, fontsize=PS["legend_fontsize"] - 4, frameon=False,
        )

    axes[0].set_ylabel("Recovery rate", fontsize=PS["label_fontsize"])
    fig.suptitle(
        f"Activation patching recovery — {model_name} "
        f"@ L_mask={level_to_show} (vis={vis:.2f})",
        fontsize=PS["suptitle_fontsize"],
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"])
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_patching_dim_means(
    payload: dict, save_path: Path,
) -> None:
    """Per-dim mean recovery curve at one canonical level (default L=4).

    Aggregates across all bins of a dim to give the headline 3-curve
    figure for the early/mid/late inflection story.
    """
    model_name = payload.get("model", "?")
    num_layers = payload["num_layers"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    levels_to_show = [2, 4, 6, 8]
    colors = plt.cm.tab10.colors

    for ax, level in zip(axes, levels_to_show):
        vis = payload["vis_per_level"].get(level) or \
              payload["vis_per_level"].get(str(level))
        for di, dim in enumerate(DIM_ORDER):
            groups = _group_image_ids_by_dim(payload, dim)
            ids = [iid for bin_ids in groups.values() for iid in bin_ids]
            layers, recov = _patching_recovery_per_group(
                payload, level, ids,
            )
            ax.plot(
                layers, recov,
                marker=PS["marker"], markersize=PS["markersize"],
                linewidth=PS["linewidth"], color=colors[di], label=dim,
            )
        ax.axhline(0, color="gray", linewidth=0.6, alpha=0.4)
        ax.axhline(1, color="gray", linewidth=0.6, alpha=0.4, linestyle=":")
        ax.set_xlabel("L_patch", fontsize=PS["label_fontsize"])
        ax.set_title(f"L_mask={level} (vis={vis:.2f})",
                     fontsize=PS["subplot_title_fontsize"])
        ax.tick_params(labelsize=PS["tick_labelsize"])
        ax.set_xticks(range(0, num_layers, max(1, num_layers // 6)))
        ax.set_xlim(-0.5, num_layers - 0.5)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=PS["legend_fontsize"] - 2,
                  frameon=False)

    axes[0].set_ylabel("Recovery rate", fontsize=PS["label_fontsize"])
    fig.suptitle(
        f"Activation patching: per-dim recovery vs L_patch — {model_name}",
        fontsize=PS["suptitle_fontsize"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"])
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_patching_heatmap(
    payload: dict, save_path: Path,
) -> None:
    """L_mask × L_patch heatmap of recovery, one panel per dim (overall).

    Shows the diagonal-ish pattern: recovery saturates at later L_patch as
    L_mask decreases. Aggregated over all images in each dim (no per-bin
    breakdown — that's the recovery_curves figure).
    """
    model_name = payload.get("model", "?")
    num_layers = payload["num_layers"]
    levels = sorted(int(k) for k in payload["correct_baseline_per_level"])

    fig, axes = plt.subplots(1, len(DIM_ORDER), figsize=(15, 5),
                             sharey=True)

    vmin, vmax = -0.2, 1.0
    im = None
    for ax, dim in zip(axes, DIM_ORDER):
        groups = _group_image_ids_by_dim(payload, dim)
        ids = [iid for bin_ids in groups.values() for iid in bin_ids]
        if not ids:
            ax.set_title(f"{dim} (empty)")
            ax.axis("off")
            continue

        matrix = np.zeros((len(levels), num_layers))
        for li, level in enumerate(levels):
            _, recov = _patching_recovery_per_group(payload, level, ids)
            matrix[li] = recov

        im = ax.imshow(
            matrix, aspect="auto", cmap="RdBu_r",
            vmin=vmin, vmax=vmax, interpolation="nearest", origin="lower",
        )
        ax.set_xticks(range(0, num_layers, max(1, num_layers // 8)))
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([
            f"L{L} ({payload['vis_per_level'].get(L, payload['vis_per_level'].get(str(L), 0)):.2f})"
            for L in levels
        ])
        ax.set_xlabel("L_patch", fontsize=PS["label_fontsize"])
        ax.set_title(dim, fontsize=PS["subplot_title_fontsize"])
        ax.tick_params(labelsize=PS["tick_labelsize"] - 2)

    axes[0].set_ylabel("L_mask (vis ratio)", fontsize=PS["label_fontsize"])
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("Recovery rate", fontsize=PS["label_fontsize"])
        cbar.ax.tick_params(labelsize=PS["tick_labelsize"] - 2)

    fig.suptitle(
        f"Patching recovery heatmap — {model_name}",
        fontsize=PS["suptitle_fontsize"],
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CAV training and ablation results.",
    )
    parser.add_argument(
        "--cavs", type=Path,
        default=Path("data/coco_subset_56/cavs/clip_L14.pt"),
    )
    parser.add_argument(
        "--ablation", type=Path,
        default=Path("results/exp3/cav_ablation_full_vis_clip_L14.json"),
    )
    parser.add_argument(
        "--degrade", type=Path,
        default=Path("results/exp3/cav_degrade_clip_L14.json"),
    )
    parser.add_argument(
        "--subspace", type=Path,
        default=Path("results/exp3/cav_subspace_ablate_clip_L14.json"),
    )
    parser.add_argument(
        "--retrieval", type=Path,
        default=Path("results/exp3/cav_subspace_retrieval_clip_L14.json"),
    )
    parser.add_argument(
        "--patch", type=Path,
        default=Path("results/exp3/cav_patch_clip_L14.pt"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/exp3"),
    )
    args = parser.parse_args()

    if args.cavs.exists():
        payload = torch.load(args.cavs, weights_only=False)
        plot_layer_concept_heatmap(
            payload, args.out_dir / "cav_layer_heatmap_clip_L14.png",
        )
        plot_dim_mean_curves(
            payload, args.out_dir / "cav_layer_mean_curve_clip_L14.png",
        )

    if args.ablation.exists():
        plot_ablation_double_diff(
            args.ablation,
            args.out_dir / "cav_ablation_double_diff_clip_L14.png",
        )
        plot_ablation_pos_neg_scatter(
            args.ablation,
            args.out_dir / "cav_ablation_pos_neg_scatter_clip_L14.png",
        )

    if args.degrade.exists():
        plot_degradation_curves(
            args.degrade,
            args.out_dir / "cav_degrade_curves_clip_L14.png",
        )
        plot_degradation_dim_means(
            args.degrade,
            args.out_dir / "cav_degrade_dim_means_clip_L14.png",
        )

    if args.subspace.exists():
        plot_subspace_ablation(
            args.subspace,
            args.out_dir / "cav_subspace_ablate_clip_L14.png",
        )

    if args.retrieval.exists():
        plot_subspace_retrieval(
            args.retrieval,
            args.out_dir / "cav_subspace_retrieval_clip_L14.png",
        )

    if args.patch.exists():
        patch_payload = torch.load(args.patch, weights_only=False)
        plot_patching_recovery_curves(
            patch_payload,
            args.out_dir / "cav_patch_recovery_clip_L14.png",
            level_to_show=4,
        )
        plot_patching_dim_means(
            patch_payload,
            args.out_dir / "cav_patch_dim_means_clip_L14.png",
        )
        plot_patching_heatmap(
            patch_payload,
            args.out_dir / "cav_patch_heatmap_clip_L14.png",
        )


if __name__ == "__main__":
    main()

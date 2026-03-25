"""Unified plotting from results.json.

Subcommands:
    combined        All encoders combined (5-subplot + individual)
    by-encoder      Per-encoder plots (lines = image types)
    similarity-diff Similarity difference (target - all) plots
    all             Run all of the above

Usage:
    uv run python plot.py all --results results/results.json
    uv run python plot.py combined --results results/results.json
    uv run python plot.py by-encoder --results results/results.json
    uv run python plot.py similarity-diff --results results/results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from src.config import (
    ENCODER_COLORS,
    ENCODER_DISPLAY_ORDER,
    IMAGE_TYPES,
    IMAGE_TYPE_ALPHA,
    IMAGE_TYPE_COLORS,
    PLOT_STYLE as PS,
    dir_to_display,
    display_to_dir,
    results_all_encoders,
    results_for_encoder,
    results_for_image_type,
)
from src.masking import get_mask_levels, get_visibility_ratio
from src.utils import (
    extract_val,
    extract_std,
    fix_json_keys,
    make_fig,
    plot_metric_vs_masking,
    plot_completion_summary,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_unified(results_path: Path) -> dict:
    """Load unified results.json and convert string keys to int."""
    with open(results_path) as f:
        data = json.load(f)
    # Handle both {"encoders": {...}} and flat format
    encoders = data.get("encoders", data)
    # Fix int keys recursively
    return {enc: {img_type: {metric: fix_json_keys(vals)
                              for metric, vals in metrics.items()}
                  for img_type, metrics in img_types.items()}
            for enc, img_types in encoders.items()}


def _load_legacy(results_dir: Path) -> dict:
    """Load from legacy per-image-type results.json files.

    Searches results/image_types/{type}/results.json and legacy dirs.
    Returns {encoder: {img_type: {task: {level: val}}}}.
    """
    merged = {}
    for img_type in IMAGE_TYPES:
        candidates = [
            results_dir / "image_types" / img_type / "results.json",
        ]
        path = None
        for c in candidates:
            if c.exists():
                path = c
                break
        if path is None:
            continue

        with open(path) as f:
            data = json.load(f)
        print(f"  Loaded {path}")

        for task in ("gestalt", "mnemonic", "semantic"):
            if task not in data:
                continue
            for enc, vals in data[task].items():
                if enc not in merged:
                    merged[enc] = {}
                if img_type not in merged[enc]:
                    merged[enc][img_type] = {}
                if task == "gestalt":
                    merged[enc][img_type]["gestalt_iou"] = fix_json_keys(vals)
                elif task == "mnemonic":
                    fixed = fix_json_keys(vals)
                    merged[enc][img_type]["mnemonic_similarity"] = fixed.get("similarity", fixed)
                    merged[enc][img_type]["mnemonic_retrieval"] = fixed.get("retrieval", {})
                elif task == "semantic":
                    fixed = fix_json_keys(vals)
                    merged[enc][img_type]["semantic_prototype"] = fixed.get("prototype_acc", fixed)
                    if "zeroshot_acc" in fixed:
                        merged[enc][img_type]["semantic_zeroshot"] = fixed["zeroshot_acc"]

    # Also load similarity_analysis.json from encoders/ dirs
    encoders_dir = results_dir / "encoders"
    if encoders_dir.exists():
        for enc_dir in sorted(encoders_dir.iterdir()):
            sim_path = enc_dir / "mnemonic" / "similarity_analysis.json"
            if not sim_path.exists():
                continue
            try:
                display = dir_to_display(enc_dir.name)
            except KeyError:
                display = enc_dir.name
            with open(sim_path) as f:
                sim_data = json.load(f)
            for img_type, sims in sim_data.items():
                if display not in merged:
                    merged[display] = {}
                if img_type not in merged[display]:
                    merged[display][img_type] = {}
                for key in ("mnemonic_target", "mnemonic_all",
                            "semantic_same_cat", "semantic_all_cat"):
                    if key in sims:
                        merged[display][img_type][f"similarity_{key}"] = fix_json_keys(sims[key])

    return merged


def load_results(results_path: Path) -> dict:
    """Load results from unified JSON or legacy directory structure."""
    if results_path.is_file():
        return _load_unified(results_path)
    elif results_path.is_dir():
        return _load_legacy(results_path)
    else:
        raise FileNotFoundError(f"Results not found: {results_path}")


# ---------------------------------------------------------------------------
# Combined plot helpers
# ---------------------------------------------------------------------------

def _get_xy(values, levels):
    x = np.array([get_visibility_ratio(L) for L in levels])
    y = np.array([extract_val(values[L]) for L in levels])
    std = np.array([extract_std(values[L]) for L in levels])
    return x, y, std


def _plot_panel(ax, panel_data, title, ylabel):
    """Plot one panel for combined figure (color=encoder, alpha=image_type)."""
    levels = get_mask_levels()
    for enc, img_type, values in panel_data:
        color = ENCODER_COLORS.get(enc, (0, 0, 0))
        alpha = IMAGE_TYPE_ALPHA.get(img_type, 1.0)
        x, y, std = _get_xy(values, levels)
        ax.plot(x, y, marker=PS["marker"], markersize=PS["markersize"],
                linewidth=PS["linewidth"], color=(*color[:3], alpha))
        if np.any(std > 0):
            ax.fill_between(x, y - std, y + std,
                            color=(*color[:3], alpha * PS["std_alpha"]))
    ax.set_xlabel("Visibility", fontsize=PS["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=PS["label_fontsize"])
    ax.set_title(title, fontsize=PS["subplot_title_fontsize"], fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=PS["tick_labelsize"], width=PS["tick_width"])
    for spine in ax.spines.values():
        spine.set_linewidth(PS["tick_width"])
    ax.grid(True, alpha=0.3)


def _make_legend_handles(encoders):
    enc_handles = [
        mlines.Line2D([], [], color=ENCODER_COLORS.get(e, "black"),
                       marker=PS["marker"], markersize=PS["markersize"],
                       linewidth=PS["linewidth"], label=e)
        for e in encoders
    ]
    type_handles = [
        mlines.Line2D([], [], color=(0.3, 0.3, 0.3, IMAGE_TYPE_ALPHA[t]),
                       marker=PS["marker"], markersize=PS["markersize"],
                       linewidth=PS["linewidth"], label=t)
        for t in IMAGE_TYPES
    ]
    return enc_handles, type_handles


# ---------------------------------------------------------------------------
# Subcommand: combined
# ---------------------------------------------------------------------------

def cmd_combined(data: dict, out_root: Path):
    """All encoders combined: 5-subplot + individual plots."""
    encoders = [e for e in ENCODER_DISPLAY_ORDER if e in data]
    if not encoders:
        encoders = sorted(data.keys())

    levels = get_mask_levels()

    # Build panel data: list of (enc, img_type, values) tuples
    def _items(metric):
        items = []
        for enc in encoders:
            for img_type in IMAGE_TYPES:
                if img_type in data.get(enc, {}) and metric in data[enc][img_type]:
                    items.append((enc, img_type, data[enc][img_type][metric]))
        return items

    panels = [
        ("Gestalt (IoU)",          "IoU",        _items("gestalt_iou")),
        ("Mnemonic (Similarity)",  "Cosine Sim", _items("mnemonic_similarity")),
        ("Mnemonic (Retrieval@1)", "Accuracy",   _items("mnemonic_retrieval")),
        ("Semantic (Prototype)",   "Accuracy",   _items("semantic_prototype")),
    ]
    zs = _items("semantic_zeroshot")
    if zs:
        panels.append(("Semantic (Zero-shot)", "Accuracy", zs))

    all_dir = results_all_encoders(root=out_root)
    all_dir.mkdir(parents=True, exist_ok=True)

    # Combined 5-subplot figure
    n = len(panels)
    fig, axes = make_fig(1, n)
    if n == 1:
        axes = [axes]
    for ax, (title, ylabel, items) in zip(axes, panels):
        _plot_panel(ax, items, title, ylabel)

    enc_handles, type_handles = _make_legend_handles(encoders)
    # Row 1: encoders, Row 2: image types — two separate legends
    leg1 = fig.legend(handles=enc_handles, loc="outside lower center",
                      ncol=len(enc_handles), fontsize=PS["legend_fontsize"],
                      bbox_to_anchor=(0.5, -0.1),
                      frameon=False)
    fig.legend(handles=type_handles, loc="outside lower center",
               ncol=len(type_handles), fontsize=PS["legend_fontsize"],
               bbox_to_anchor=(0.5, -0.16),
               frameon=False)
    fig.add_artist(leg1)
    fig.suptitle("Fragment Completion — All Encoders",
                 fontsize=PS["suptitle_fontsize"], fontweight="bold")
    save = all_dir / "completion_summary.png"
    fig.savefig(save, dpi=PS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save}")

    # Individual plots
    fpath_map = [
        "gestalt/gestalt_iou",
        "mnemonic/mnemonic_similarity",
        "mnemonic/mnemonic_retrieval",
        "semantic/semantic_prototype",
        "semantic/semantic_zeroshot",
    ]
    for (title, ylabel, items), fpath in zip(panels, fpath_map):
        fig, ax = make_fig(1, 1)
        _plot_panel(ax, items, title, ylabel)
        enc_h, type_h = _make_legend_handles(encoders)
        leg1 = fig.legend(handles=enc_h, loc="outside lower center",
                          ncol=len(enc_h), fontsize=PS["legend_fontsize"],
                          bbox_to_anchor=(0.5, -0.1),
                          frameon=False)
        fig.legend(handles=type_h, loc="outside lower center",
                   ncol=len(type_h), fontsize=PS["legend_fontsize"],
                   bbox_to_anchor=(0.5, -0.16),
                   frameon=False)
        fig.add_artist(leg1)
        sp = all_dir / f"{fpath}.png"
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sp, dpi=PS["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {sp}")


# ---------------------------------------------------------------------------
# Subcommand: by-encoder
# ---------------------------------------------------------------------------

def cmd_by_encoder(data: dict, out_root: Path):
    """Per-encoder plots: lines = image types."""
    for enc in sorted(data.keys()):
        enc_dir = results_for_encoder(enc, root=out_root)
        enc_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  === {enc} -> {enc_dir} ===")

        enc_data = data[enc]  # {img_type: {metric: {level: val}}}

        # Gestalt
        gestalt = {it: enc_data[it]["gestalt_iou"]
                   for it in IMAGE_TYPES if it in enc_data and "gestalt_iou" in enc_data[it]}
        if gestalt:
            plot_metric_vs_masking(gestalt, "IoU", f"{enc} — Gestalt (IoU)",
                                   enc_dir / "gestalt" / "gestalt_iou.png", colors=IMAGE_TYPE_COLORS)

        # Mnemonic
        mnem_sim = {it: enc_data[it]["mnemonic_similarity"]
                    for it in IMAGE_TYPES if it in enc_data and "mnemonic_similarity" in enc_data[it]}
        mnem_ret = {it: enc_data[it]["mnemonic_retrieval"]
                    for it in IMAGE_TYPES if it in enc_data and "mnemonic_retrieval" in enc_data[it]}
        if mnem_sim:
            plot_metric_vs_masking(mnem_sim, "Cosine Similarity",
                                   f"{enc} — Mnemonic (Similarity)",
                                   enc_dir / "mnemonic" / "mnemonic_similarity.png", colors=IMAGE_TYPE_COLORS)
        if mnem_ret:
            plot_metric_vs_masking(mnem_ret, "Top-1 Accuracy",
                                   f"{enc} — Mnemonic (Retrieval)",
                                   enc_dir / "mnemonic" / "mnemonic_retrieval.png", colors=IMAGE_TYPE_COLORS)

        # Semantic
        sem_proto = {it: enc_data[it]["semantic_prototype"]
                     for it in IMAGE_TYPES if it in enc_data and "semantic_prototype" in enc_data[it]}
        sem_zs = {it: enc_data[it]["semantic_zeroshot"]
                  for it in IMAGE_TYPES if it in enc_data and "semantic_zeroshot" in enc_data[it]}
        if sem_proto:
            plot_metric_vs_masking(sem_proto, "Accuracy",
                                   f"{enc} — Semantic (Prototype)",
                                   enc_dir / "semantic" / "semantic_prototype.png", colors=IMAGE_TYPE_COLORS)
        if sem_zs:
            plot_metric_vs_masking(sem_zs, "Accuracy",
                                   f"{enc} — Semantic (Zero-shot)",
                                   enc_dir / "semantic" / "semantic_zeroshot.png", colors=IMAGE_TYPE_COLORS)

        # Summary
        # Reconstruct mnemonic/semantic dicts expected by plot_completion_summary
        mnem_combined = {}
        sem_combined = {}
        for it in IMAGE_TYPES:
            if it in enc_data:
                if "mnemonic_similarity" in enc_data[it]:
                    mnem_combined[it] = {
                        "similarity": enc_data[it]["mnemonic_similarity"],
                        "retrieval": enc_data[it].get("mnemonic_retrieval", {}),
                    }
                sem_entry = {}
                if "semantic_prototype" in enc_data[it]:
                    sem_entry["prototype_acc"] = enc_data[it]["semantic_prototype"]
                if "semantic_zeroshot" in enc_data[it]:
                    sem_entry["zeroshot_acc"] = enc_data[it]["semantic_zeroshot"]
                if sem_entry:
                    sem_combined[it] = sem_entry

        plot_completion_summary(
            gestalt or None, mnem_combined or None, sem_combined or None,
            enc_dir / "completion_summary.png", colors=IMAGE_TYPE_COLORS,
        )


# ---------------------------------------------------------------------------
# Subcommand: by-image-type
# ---------------------------------------------------------------------------

def cmd_by_image_type(data: dict, out_root: Path):
    """Per-image-type plots: lines = encoders."""
    # Pivot: data is {encoder: {img_type: {metric: ...}}}
    # We need {img_type: {encoder: {metric: ...}}}
    by_type: dict[str, dict[str, dict]] = {}
    for enc, enc_data in data.items():
        for img_type, metrics in enc_data.items():
            by_type.setdefault(img_type, {})[enc] = metrics

    for img_type in IMAGE_TYPES:
        if img_type not in by_type:
            continue
        type_data = by_type[img_type]  # {encoder: {metric: {level: val}}}
        out_dir = results_for_image_type(img_type, root=out_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  === {img_type} -> {out_dir} ===")

        # Gestalt
        gestalt = {enc: type_data[enc]["gestalt_iou"]
                   for enc in type_data if "gestalt_iou" in type_data[enc]}
        if gestalt:
            plot_metric_vs_masking(gestalt, "IoU", f"{img_type} — Gestalt (IoU)",
                                   out_dir / "gestalt" / "gestalt_iou.png")
            sil_data = {}
            for enc, vals in gestalt.items():
                first = next(iter(vals.values()))
                if isinstance(first, dict) and "silhouette_mean" in first:
                    sil_data[enc] = {
                        L: {"mean": v["silhouette_mean"], "std": v["silhouette_std"]}
                        for L, v in vals.items()
                    }
            if sil_data:
                plot_metric_vs_masking(sil_data, "Silhouette Score",
                                       f"{img_type} — Gestalt (Cluster Separation)",
                                       out_dir / "gestalt" / "gestalt_silhouette.png")

        # Mnemonic
        mnem_sim = {enc: type_data[enc]["mnemonic_similarity"]
                    for enc in type_data if "mnemonic_similarity" in type_data[enc]}
        mnem_ret = {enc: type_data[enc]["mnemonic_retrieval"]
                    for enc in type_data if "mnemonic_retrieval" in type_data[enc]}
        if mnem_sim:
            plot_metric_vs_masking(mnem_sim, "Cosine Similarity",
                                   f"{img_type} — Mnemonic (Similarity)",
                                   out_dir / "mnemonic" / "mnemonic_similarity.png")
        if mnem_ret:
            plot_metric_vs_masking(mnem_ret, "Top-1 Accuracy",
                                   f"{img_type} — Mnemonic (Retrieval)",
                                   out_dir / "mnemonic" / "mnemonic_retrieval.png")

        # Semantic
        sem_proto = {enc: type_data[enc]["semantic_prototype"]
                     for enc in type_data if "semantic_prototype" in type_data[enc]}
        sem_zs = {enc: type_data[enc]["semantic_zeroshot"]
                  for enc in type_data if "semantic_zeroshot" in type_data[enc]}
        if sem_proto:
            plot_metric_vs_masking(sem_proto, "Accuracy",
                                   f"{img_type} — Semantic (Prototype)",
                                   out_dir / "semantic" / "semantic_prototype.png")
        if sem_zs:
            plot_metric_vs_masking(sem_zs, "Accuracy",
                                   f"{img_type} — Semantic (Zero-shot)",
                                   out_dir / "semantic" / "semantic_zeroshot.png")

        # Summary
        mnem_combined = {}
        sem_combined = {}
        for enc in type_data:
            m = type_data[enc]
            if "mnemonic_similarity" in m:
                mnem_combined[enc] = {
                    "similarity": m["mnemonic_similarity"],
                    "retrieval": m.get("mnemonic_retrieval", {}),
                }
            sem_entry = {}
            if "semantic_prototype" in m:
                sem_entry["prototype_acc"] = m["semantic_prototype"]
            if "semantic_zeroshot" in m:
                sem_entry["zeroshot_acc"] = m["semantic_zeroshot"]
            if sem_entry:
                sem_combined[enc] = sem_entry

        plot_completion_summary(
            gestalt or None, mnem_combined or None, sem_combined or None,
            out_dir / "completion_summary.png",
        )


# ---------------------------------------------------------------------------
# Subcommand: similarity-diff
# ---------------------------------------------------------------------------

def cmd_similarity_diff(data: dict, out_root: Path):
    """Similarity difference plots: (target - all) per image type."""
    levels = get_mask_levels()

    for img_type in IMAGE_TYPES:
        fig, (ax1, ax2) = make_fig(1, 2, sharey=False)

        for enc in sorted(data.keys()):
            if img_type not in data[enc]:
                continue
            d = data[enc][img_type]

            # Check if similarity data exists
            has_mnem = ("similarity_mnemonic_target" in d and
                        "similarity_mnemonic_all" in d)
            has_sem = ("similarity_semantic_same_cat" in d and
                       "similarity_semantic_all_cat" in d)
            if not has_mnem and not has_sem:
                continue

            color = ENCODER_COLORS.get(enc, None)

            if has_mnem:
                mn_target = [extract_val(d["similarity_mnemonic_target"][L]) for L in levels]
                mn_all = [extract_val(d["similarity_mnemonic_all"][L]) for L in levels]
                mn_diff = [t - a for t, a in zip(mn_target, mn_all)]
                mn_t_std = [extract_std(d["similarity_mnemonic_target"][L]) for L in levels]
                mn_a_std = [extract_std(d["similarity_mnemonic_all"][L]) for L in levels]
                mn_diff_std = [np.sqrt(s1**2 + s2**2) for s1, s2 in zip(mn_t_std, mn_a_std)]

                ax1.plot(levels, mn_diff, "o-", label=enc, color=color,
                         linewidth=PS["linewidth"], markersize=PS["markersize"])
                ax1.fill_between(levels,
                                 [m - s for m, s in zip(mn_diff, mn_diff_std)],
                                 [m + s for m, s in zip(mn_diff, mn_diff_std)],
                                 alpha=PS["std_alpha"], color=color)

            if has_sem:
                sem_same = [extract_val(d["similarity_semantic_same_cat"][L]) for L in levels]
                sem_all = [extract_val(d["similarity_semantic_all_cat"][L]) for L in levels]
                sem_diff = [t - a for t, a in zip(sem_same, sem_all)]
                sem_s_std = [extract_std(d["similarity_semantic_same_cat"][L]) for L in levels]
                sem_a_std = [extract_std(d["similarity_semantic_all_cat"][L]) for L in levels]
                sem_diff_std = [np.sqrt(s1**2 + s2**2) for s1, s2 in zip(sem_s_std, sem_a_std)]

                ax2.plot(levels, sem_diff, "o-", label=enc, color=color,
                         linewidth=PS["linewidth"], markersize=PS["markersize"])
                ax2.fill_between(levels,
                                 [m - s for m, s in zip(sem_diff, sem_diff_std)],
                                 [m + s for m, s in zip(sem_diff, sem_diff_std)],
                                 alpha=PS["std_alpha"], color=color)

        xtick_labels = [f"L{L}\n({get_visibility_ratio(L):.0%})" for L in levels]
        for ax, title in [(ax1, "Mnemonic (target - all)"),
                          (ax2, "Semantic (same_cat - all_cat)")]:
            ax.set_xlabel("Fragmentation Level", fontsize=PS["label_fontsize"])
            ax.set_ylabel("Similarity Difference", fontsize=PS["label_fontsize"])
            ax.set_title(title, fontsize=PS["subplot_title_fontsize"], fontweight="bold")
            ax.set_xticks(levels)
            ax.set_xticklabels(xtick_labels, fontsize=PS["tick_labelsize"])
            ax.tick_params(width=PS["tick_width"])
            for spine in ax.spines.values():
                spine.set_linewidth(PS["tick_width"])
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # One shared legend below both subplots
        handles, labels = ax1.get_legend_handles_labels()
        if not handles:
            handles, labels = ax2.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center",
                       ncol=len(labels), fontsize=PS["legend_fontsize"],
                       frameon=True)
        fig.suptitle(f"Similarity Difference — {img_type}",
                     fontsize=PS["suptitle_fontsize"], fontweight="bold")

        # Save into all_encoders/ directory alongside other combined plots
        save_dir = results_all_encoders(root=out_root)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"similarity_diff_{img_type}.png"
        fig.savefig(save_path, dpi=PS["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot fragment completion results")
    sub = parser.add_subparsers(dest="command", required=True)

    # Common args
    for name in ("combined", "by-encoder", "by-image-type", "similarity-diff", "all"):
        p = sub.add_parser(name)
        p.add_argument("--results", type=str, default="results/results.json",
                        help="Path to results.json or results directory")
        p.add_argument("--out-dir", type=str, default="results",
                        help="Output directory root")

    args = parser.parse_args()
    results_path = Path(args.results)
    out_root = Path(args.out_dir)

    print(f"Loading results from {results_path}...")
    data = load_results(results_path)
    print(f"  Found {len(data)} encoders: {sorted(data.keys())}")

    cmd = args.command
    if cmd in ("combined", "all"):
        print("\n--- Combined plots ---")
        cmd_combined(data, out_root)
    if cmd in ("by-encoder", "all"):
        print("\n--- Per-encoder plots ---")
        cmd_by_encoder(data, out_root)
    if cmd in ("by-image-type", "all"):
        print("\n--- Per-image-type plots ---")
        cmd_by_image_type(data, out_root)
    if cmd in ("similarity-diff", "all"):
        print("\n--- Similarity diff plots ---")
        cmd_similarity_diff(data, out_root)

    print("\nDone!")


if __name__ == "__main__":
    main()

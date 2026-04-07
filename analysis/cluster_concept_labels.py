#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cluster_concept_labels.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Cluster open-ended concept labels into canonical bins.

Reads data/coco_subset_56/concept_labels.json (open-ended labels per image
across 4 dimensions: color, material, scene, function) and clusters the
unique values per dimension into K canonical bins using sentence-transformer
embeddings + agglomerative clustering with cosine distance.

Output: data/coco_subset_56/concept_clusters.json
  {
    "color":    {"raw_to_cluster": {raw: cluster_name, ...}, "cluster_to_raws": {...}, "k": 12},
    "material": {...},
    "function": {...}
  }
  (scene is dropped per the salience analysis: ~17% high-salience only)

Usage:
    uv run python -m analysis.cluster_concept_labels
    uv run python -m analysis.cluster_concept_labels --k-color 12 --k-material 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


# Default cluster counts per dimension. Tuned to give a reasonable
# semantic granularity without over-fragmenting.
DEFAULT_K = {
    "color": 16,
    "material": 16,
    "function": 14,
}

# Skip scene by default (salience analysis showed only ~17% high-salience).
SKIP_DIMS = {"scene"}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Template per dimension to give the embedding model more context.
# Short single-word labels collapse together without context;
# wrapping in a sentence pulls dimension-specific semantics out.
TEMPLATES = {
    "color": "the color of the object is {value}",
    "material": "the object is made of {value}",
    "function": "the object is used for {value}",
}


def load_labels(path: Path) -> list[dict]:
    """Load aggregated open-ended concept labels."""
    with open(path) as f:
        return json.load(f)


def collect_unique_values(
    entries: list[dict], dim: str, min_salience: str = "low",
) -> tuple[list[str], dict[str, int]]:
    """Collect unique values for a dimension, weighted by frequency.

    Args:
        entries: List of label dicts.
        dim: Dimension name (color, material, ...).
        min_salience: Minimum salience to include. "none" entries are
            always dropped (no information). "low" includes everything else.

    Returns:
        (unique_values, value_to_count) — values sorted by frequency desc.
    """
    counts: Counter[str] = Counter()
    for e in entries:
        if e[dim]["salience"] == "none":
            continue
        if min_salience == "medium" and e[dim]["salience"] == "low":
            continue
        if min_salience == "high" and e[dim]["salience"] != "high":
            continue
        counts[e[dim]["value"]] += 1
    values = [v for v, _ in counts.most_common()]
    return values, dict(counts)


def cluster_values(
    values: list[str],
    counts: dict[str, int],
    k: int,
    model: SentenceTransformer,
    template: str | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Cluster a list of free-text values into k bins.

    Pipeline:
        1. Embed each value (optionally wrapped in a template) with
           sentence-transformer.
        2. K-means on L2-normalized embeddings (equivalent to spherical
           k-means / cosine k-means, naturally balanced).
        3. Each cluster's name = the value with the highest count in it
           (most frequent representative).

    Args:
        values: Unique free-text values.
        counts: Frequency per value.
        k: Number of target clusters.
        model: Loaded sentence-transformer model.
        template: Optional format string with {value} placeholder, used
            to wrap short labels in a sentence for better embedding.

    Returns:
        (raw_to_cluster, cluster_to_raws)
    """
    if len(values) <= k:
        return ({v: v for v in values}, {v: [v] for v in values})

    texts = (
        [template.format(value=v) for v in values] if template else values
    )
    embeds = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # K-means on L2-normalized embeddings = spherical k-means (cosine).
    # Naturally produces balanced clusters, unlike average-linkage
    # agglomerative which chains everything into one giant cluster.
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = clf.fit_predict(embeds)

    cluster_to_raws: dict[int, list[str]] = defaultdict(list)
    for v, c in zip(values, labels):
        cluster_to_raws[int(c)].append(v)

    # Pick representative name = most frequent value in cluster
    cluster_names: dict[int, str] = {}
    for c, members in cluster_to_raws.items():
        rep = max(members, key=lambda v: counts.get(v, 0))
        cluster_names[c] = rep

    raw_to_cluster: dict[str, str] = {
        v: cluster_names[int(c)] for v, c in zip(values, labels)
    }
    cluster_to_raws_named: dict[str, list[str]] = {
        cluster_names[c]: sorted(members, key=lambda v: -counts.get(v, 0))
        for c, members in cluster_to_raws.items()
    }
    return raw_to_cluster, cluster_to_raws_named


def report_clusters(
    dim: str,
    cluster_to_raws: dict[str, list[str]],
    counts: dict[str, int],
    top_k_show: int = 6,
) -> None:
    """Print a human-readable cluster report for one dimension."""
    print(f"\n=== {dim.upper()} — {len(cluster_to_raws)} clusters ===")
    cluster_totals: list[tuple[str, int, list[str]]] = []
    for cname, members in cluster_to_raws.items():
        total = sum(counts.get(m, 0) for m in members)
        cluster_totals.append((cname, total, members))
    cluster_totals.sort(key=lambda t: -t[1])
    for cname, total, members in cluster_totals:
        head = ", ".join(
            f"{m} ({counts.get(m, 0)})" for m in members[:top_k_show]
        )
        more = f" ... +{len(members) - top_k_show} more" if len(members) > top_k_show else ""
        print(f"  [{cname}] (n={total}, {len(members)} raw): {head}{more}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster open-ended concept labels into canonical bins.",
    )
    parser.add_argument(
        "--labels", type=Path,
        default=Path("data/coco_subset_56/concept_labels.json"),
        help="Path to aggregated open-ended labels JSON.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/coco_subset_56/concept_clusters.json"),
        help="Output path for cluster mapping JSON.",
    )
    parser.add_argument("--k-color", type=int, default=DEFAULT_K["color"])
    parser.add_argument("--k-material", type=int, default=DEFAULT_K["material"])
    parser.add_argument("--k-function", type=int, default=DEFAULT_K["function"])
    parser.add_argument(
        "--min-salience", choices=["low", "medium", "high"], default="low",
        help="Drop labels below this salience before clustering.",
    )
    args = parser.parse_args()

    print(f"Loading labels from {args.labels}")
    entries = load_labels(args.labels)
    print(f"  {len(entries)} entries")

    print(f"\nLoading sentence-transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    k_per_dim = {
        "color": args.k_color,
        "material": args.k_material,
        "function": args.k_function,
    }

    output: dict[str, dict] = {}
    for dim in ("color", "material", "function"):
        if dim in SKIP_DIMS:
            continue
        values, counts = collect_unique_values(entries, dim, args.min_salience)
        print(f"\n[{dim}] {len(values)} unique values "
              f"(total {sum(counts.values())} labels), "
              f"clustering into k={k_per_dim[dim]}")

        raw_to_cluster, cluster_to_raws = cluster_values(
            values, counts, k_per_dim[dim], model,
            template=TEMPLATES.get(dim),
        )
        report_clusters(dim, cluster_to_raws, counts)

        output[dim] = {
            "k": k_per_dim[dim],
            "min_salience": args.min_salience,
            "n_unique": len(values),
            "raw_to_cluster": raw_to_cluster,
            "cluster_to_raws": cluster_to_raws,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved cluster mapping to {args.out}")


if __name__ == "__main__":
    main()

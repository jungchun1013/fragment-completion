#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : fix_concept_clusters.py
# Author : Jung-Chun Liu
# Email  : jungchun@umich.edu
# Date   : 04-07-2026
#
# This file is part of fragment-completion.
# Distributed under terms of the MIT license.
"""Manual surgical fixes to concept_clusters.json.

Reads data/coco_subset_56/concept_clusters.json (output of
cluster_concept_labels.py) and applies hand-coded reassignments to fix
the boundary issues that K-means missed:

- color: extract [white] and [multicolor] from [black]; move 'stainless steel'
  out of [cream] into [silver]; extract [pink] from [cream]
- material: move 'plastic screen' out of [hide]; pull 'banh mi' / 'paint'
  out of [wood]
- function: split [washing] into [washing]/[storage]/[reading]; merge the
  bizarre [military aircraft] grab-bag into [personal transport]; move
  'picnic table', 'pitcher', boat-locomotion strays to their proper homes

Usage:
    uv run python -m analysis.fix_concept_clusters
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


CLUSTERS_PATH = Path("data/coco_subset_56/concept_clusters.json")
LABELS_PATH = Path("data/coco_subset_56/concept_labels.json")


# (raw_value, new_cluster_name) overrides per dimension. Anything not
# listed keeps its original cluster.
OVERRIDES: dict[str, dict[str, str]] = {
    "color": {
        # Extract [white] from [black] cluster
        "white": "white",
        "off-white": "white",
        "dirty white": "white",
        "clear": "white",
        "clear transparent": "white",
        "clear white": "white",
        # Extract [multicolor] from [black] cluster
        "multicolor": "multicolor",
        "mixed": "multicolor",
        "white multicolor": "multicolor",
        "black multicolor": "multicolor",
        "colorful": "multicolor",
        "calico": "multicolor",
        "multicolor floral": "multicolor",
        # 'stainless steel' is a metal finish — belongs with silver
        "stainless steel": "silver",
        # Extract [pink] from [cream] cluster
        "pink": "pink",
        "pink white": "pink",
        "cream pink": "pink",
    },
    "material": {
        # 'plastic screen' / 'screen plastic' wrongly landed in [hide]
        "plastic screen": "plastic",
        "screen plastic": "plastic",
        # Strays in [wood] cluster
        "banh mi": "sausage bread",
        "paint": "unclear",
    },
    "function": {
        # Extract [storage] from [washing]
        "storage": "storage",
        "storage container": "storage",
        "wearable storage": "storage",
        "cold storage": "storage",
        "hiking storage": "storage",
        "takeout container": "storage",
        "container": "storage",
        "carry bag": "storage",
        "laptop carrying": "storage",
        "equipment bag": "storage",
        # Extract [reading] from [washing]
        "reading": "reading",
        "instructional book": "reading",
        "game manual": "reading",
        "printed media": "reading",
        "knowledge": "reading",
        "information": "reading",
        # Merge bizarre [military aircraft] grab-bag into [personal transport]
        "military aircraft": "personal transport",
        "pickup truck": "personal transport",
        "passenger vehicle": "personal transport",
        "box truck": "personal transport",
        "mountain biking": "personal transport",
        "road marking": "personal transport",
        "luggage frame": "personal transport",
        "camper trailer": "personal transport",
        "bicycle": "personal transport",
        "utility trailer": "personal transport",
        "police utility": "personal transport",
        "excavator": "personal transport",
        "tricycle": "personal transport",
        "decorated truck": "personal transport",
        "armored vehicle": "personal transport",
        "dump truck": "personal transport",
        "minivan": "personal transport",
        "garbage truck": "personal transport",
        "fire truck": "personal transport",
        # Picnic table doesn't belong with [dessert]
        "picnic table": "seating",
        # Pitcher belongs with drinkware, not water sport
        "pitcher": "drinkware",
        # Boat-locomotion strays in [water sport]
        "steam locomotion": "personal transport",
        "tour coach": "personal transport",
    },
}


def _build_counts(entries: list[dict], dim: str) -> dict[str, int]:
    """Recompute raw value frequencies for a dimension (excludes 'none')."""
    counts: Counter[str] = Counter()
    for e in entries:
        if e[dim]["salience"] == "none":
            continue
        counts[e[dim]["value"]] += 1
    return dict(counts)


def _apply_overrides(
    raw_to_cluster: dict[str, str],
    overrides: dict[str, str],
    counts: dict[str, int],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Apply overrides and rebuild cluster_to_raws.

    Returns:
        (raw_to_cluster_fixed, cluster_to_raws_sorted_by_count)
    """
    fixed = dict(raw_to_cluster)
    for raw, new_cluster in overrides.items():
        if raw in fixed:
            fixed[raw] = new_cluster

    grouped: dict[str, list[str]] = defaultdict(list)
    for raw, c in fixed.items():
        grouped[c].append(raw)

    cluster_to_raws = {
        c: sorted(members, key=lambda v: -counts.get(v, 0))
        for c, members in grouped.items()
    }
    return fixed, cluster_to_raws


def _report(dim: str, cluster_to_raws: dict[str, list[str]],
            counts: dict[str, int], top_k_show: int = 6) -> None:
    """Print per-cluster summary sorted by total count."""
    print(f"\n=== {dim.upper()} — {len(cluster_to_raws)} clusters ===")
    rows: list[tuple[str, int, list[str]]] = []
    for cname, members in cluster_to_raws.items():
        total = sum(counts.get(m, 0) for m in members)
        rows.append((cname, total, members))
    rows.sort(key=lambda r: -r[1])
    for cname, total, members in rows:
        head = ", ".join(
            f"{m} ({counts.get(m, 0)})" for m in members[:top_k_show]
        )
        more = f" ... +{len(members) - top_k_show} more" if len(members) > top_k_show else ""
        print(f"  [{cname}] (n={total}, {len(members)} raw): {head}{more}")


def main() -> None:
    print(f"Loading {CLUSTERS_PATH}")
    with open(CLUSTERS_PATH) as f:
        data = json.load(f)

    print(f"Loading {LABELS_PATH}")
    with open(LABELS_PATH) as f:
        entries = json.load(f)

    output = {}
    for dim in ("color", "material", "function"):
        if dim not in data:
            continue

        counts = _build_counts(entries, dim)
        old_raw_to_cluster = data[dim]["raw_to_cluster"]
        overrides = OVERRIDES.get(dim, {})

        # Sanity: warn about overrides whose raw values aren't in the data
        missing = [r for r in overrides if r not in old_raw_to_cluster]
        if missing:
            print(f"  [{dim}] WARNING — overrides for non-existent values: {missing}")

        new_raw_to_cluster, new_cluster_to_raws = _apply_overrides(
            old_raw_to_cluster, overrides, counts,
        )

        _report(dim, new_cluster_to_raws, counts)

        output[dim] = {
            **data[dim],
            "k": len(new_cluster_to_raws),
            "raw_to_cluster": new_raw_to_cluster,
            "cluster_to_raws": new_cluster_to_raws,
            "manual_fixes_applied": len(overrides),
        }

    with open(CLUSTERS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved fixed clusters to {CLUSTERS_PATH}")


if __name__ == "__main__":
    main()

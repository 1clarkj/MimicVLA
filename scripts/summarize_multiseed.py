"""Summarise 5-seed policy evaluations with start-level bootstrap intervals.

Reads outputs/policy_eval_multiseed/<policy>/<set>_seed<k>/summary.json.
Each start state's success probability is its mean over seeds; the 95%
interval resamples start states (not rollouts), since rollouts from the same
start are correlated.

    python scripts/summarize_multiseed.py [--markdown]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from holdout_split import split_indices


ROOT = Path("outputs/policy_eval_multiseed")
DATASET = Path("outputs/datasets/usb_15x7_no_funnel_full_v2")
ORDER = ["raw_10k", "raw_20k", "lowpass_20k", "lowpass_20k_zero_noise", "dsrl_v1", "dsrl_v2",
         "lowpass_recovery_20k"]


def load(policy: str, split: str) -> dict[int, list[dict]]:
    by_start: dict[int, list[dict]] = {}
    for path in sorted((ROOT / policy).glob(f"{split}_seed*/summary.json")):
        for row in json.loads(path.read_text())["results"]:
            by_start.setdefault(row["episode_index"], []).append(row)
    return by_start


def interval(per_start: np.ndarray, rng, draws: int = 10_000) -> tuple[float, float, float]:
    samples = rng.choice(per_start, size=(draws, len(per_start))).mean(1)
    return per_start.mean(), *np.percentile(samples, [2.5, 97.5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    holdout = set(split_indices(DATASET)["holdout"])
    rng = np.random.default_rng(0)
    groups = {"train64": lambda i: i not in holdout, "holdout8": lambda i: i in holdout}
    rows = []
    for policy in ORDER:
        all72, excluded = load(policy, "all72"), load(policy, "excluded55")
        if not all72:
            continue
        row = {"policy": policy, "seeds": len(next(iter(all72.values())))}
        for name, starts in [("train64", {i: r for i, r in all72.items() if groups["train64"](i)}),
                             ("holdout8", {i: r for i, r in all72.items() if groups["holdout8"](i)}),
                             ("excluded55", excluded)]:
            if not starts:
                continue
            p = np.array([np.mean([r["success"] for r in rs]) for rs in starts.values()])
            row[name] = interval(p, rng)
            row[name + "_collision"] = float(np.mean(
                [r["first_bad_collision_step"] is not None for rs in starts.values() for r in rs]))
            row[name + "_force"] = float(np.median(
                [r["peak_wall_force_n"] for rs in starts.values() for r in rs]))
        rows.append(row)

    def cell(v):
        return f"{100 * v[0]:.0f}% [{100 * v[1]:.0f}, {100 * v[2]:.0f}]"

    if args.markdown:
        print("| Policy | Training starts (64) | Holdout (8) | Excluded starts (55) | "
              "Deep-collision rollouts (64 / 55) | Median peak force (64 / 55) |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            print(f"| `{r['policy']}` | {cell(r['train64'])} | {cell(r['holdout8'])} | "
                  f"{cell(r['excluded55'])} | {100 * r['train64_collision']:.0f}% / "
                  f"{100 * r['excluded55_collision']:.0f}% | {r['train64_force']:.0f} / "
                  f"{r['excluded55_force']:.0f} N |")
    else:
        json.dump(rows, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()

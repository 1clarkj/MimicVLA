"""Tables and figures for the data-scaling, 2 x 2 and denoising-step experiments.

Reads the multi-seed evaluations in outputs/policy_eval_multiseed and the
latency benchmarks in outputs/latency, writes PNG figures to
documentation/figures and prints markdown tables. Success per start is its
mean over seeds; intervals are 95% bootstraps over start states.

    PYTHONPATH=scripts python scripts/summarize_overnight.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from holdout_split import split_indices, train_subset
from summarize_multiseed import interval, load


DATASET = Path("outputs/datasets/usb_15x7_no_funnel_full_v2")
FIGURES = Path("documentation/figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RNG = np.random.default_rng(0)


def rate(by_start: dict[int, list[dict]], starts=None):
    """(mean, lo, hi, n_starts, seeds) over the given starts, or None."""
    keys = [k for k in by_start if starts is None or k in starts]
    if not keys:
        return None
    p = np.array([np.mean([r["success"] for r in by_start[k]]) for k in keys])
    return (*interval(p, RNG), len(keys), min(len(by_start[k]) for k in keys))


def force(by_start, starts=None) -> float:
    return float(np.median([r["peak_wall_force_n"] for k, rs in by_start.items()
                            if starts is None or k in starts for r in rs]))


def collisions(by_start, starts=None) -> float:
    return float(np.mean([r["first_bad_collision_step"] is not None
                          for k, rs in by_start.items() if starts is None or k in starts
                          for r in rs]))


def pct(v) -> str:
    return "–" if v is None else f"{100 * v[0]:.0f}% [{100 * v[1]:.0f}, {100 * v[2]:.0f}]"


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def scaling(holdout: set[int]) -> list[dict]:
    rows = []
    points = [(8, "raw_n8_10k", train_subset(DATASET, 8)),
              (16, "raw_n16_10k", train_subset(DATASET, 16)),
              (32, "raw_n32_10k", train_subset(DATASET, 32)),
              (64, "raw_10k", split_indices(DATASET)["train"])]
    mirror_manifest = Path("outputs/datasets/usb_15x7_no_funnel_mirror_v1/capture_manifest.json")
    mirrored = 0
    if mirror_manifest.exists():
        mirrored = sum(e["mirrored"] and e["split"] == "train" for e in
                       json.loads(mirror_manifest.read_text())["episodes_manifest"])
    points.append((64 + mirrored, "raw_mirror_10k", split_indices(DATASET)["train"]))
    for demos, policy, trained in points:
        all72, excluded = load(policy, "all72"), load(policy, "excluded55")
        if not all72 and not excluded:
            continue
        trained = set(trained)
        rows.append({"demos": demos, "policy": policy, "mirror": policy == "raw_mirror_10k",
                     "trained": rate(all72, trained),
                     "unseen_dataset": rate(all72, set(range(72)) - trained),
                     "excluded": rate(excluded),
                     "excluded_force": force(excluded) if excluded else None,
                     "excluded_collisions": collisions(excluded) if excluded else None})
    return rows


def plot_scaling(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    series = [("trained", "Starts it trained on", BLUE, "o"),
              ("unseen_dataset", "Unseen dataset starts", AQUA, "s"),
              ("excluded", "55 excluded-recording starts", ORANGE, "D")]
    base = [r for r in rows if not r["mirror"]]
    mirror = [r for r in rows if r["mirror"]]
    offsets = {"trained": (-36, 3), "unseen_dataset": (9, -2), "excluded": (9, -10)}
    shifts = {"trained": 0.95, "unseen_dataset": 1.0, "excluded": 1.05}
    for key, name, colour, marker in series:
        pts = [(r["demos"], r[key]) for r in base if r[key] is not None]
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = np.array([p[1][:3] for p in pts]) * 100
        ax.fill_between(x, y[:, 1], y[:, 2], color=colour, alpha=.12, linewidth=0)
        ax.plot(x, y[:, 0], color=colour, linewidth=2, marker=marker, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=name)
        ax.annotate(f"{y[-1, 0]:.0f}%", (x[-1], y[-1, 0]), xytext=offsets[key],
                    textcoords="offset points", color=INK, fontsize=9)
        for r in mirror:
            if r[key] is not None:
                v = np.array(r[key][:3]) * 100
                ax.errorbar([r["demos"] * shifts[key]], [v[0]], yerr=[[v[0] - v[1]], [v[2] - v[0]]],
                            color=colour, marker=marker, markersize=8, mfc="white",
                            mew=2, linewidth=1.2, capsize=3)
    if mirror:
        ax.annotate(f"64 + {mirror[0]['demos'] - 64} mirrored\n(hollow markers)",
                    (mirror[0]["demos"], 4), ha="center", color=MUTED, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ticks = sorted({r["demos"] for r in rows})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("Human demonstrations in training (raw labels, 10k steps)", color=MUTED)
    ax.set_ylim(0, 100)
    style(ax, "Data scaling: closed-loop insertion success", "Success rate (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(.01, .01, "5 seeds per start; bands and bars: 95% bootstrap over start states. Unseen dataset starts:\n56 at 16 demos, 40 at 32, the 8 held-out recordings at 64 and 110",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(FIGURES / "data_scaling.png")
    plt.close(fig)


def two_by_two(holdout: set[int]) -> list[dict]:
    train = set(range(72)) - holdout
    cells = [("raw", "none", "raw_10k"), ("raw", "gentle recovery", "recovery_v2_10k"),
             ("low-pass", "none", "lowpass_20k"),
             ("low-pass", "gentle recovery", "lowpass_recovery_20k")]
    rows = []
    for labels, recovery, policy in cells:
        all72, excluded = load(policy, "all72"), load(policy, "excluded55")
        if not all72:
            continue
        rows.append({"labels": labels, "recovery": recovery, "policy": policy,
                     "train64": rate(all72, train), "holdout8": rate(all72, holdout),
                     "excluded": rate(excluded),
                     "collisions": (collisions(all72, train), collisions(excluded)),
                     "force": (force(all72, train), force(excluded))})
    return rows


def plot_two_by_two(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    groups = [("train64", "64 training starts"), ("excluded", "55 excluded-recording starts")]
    colours = [BLUE, ORANGE]
    width = .38
    for g, ((key, name), colour) in enumerate(zip(groups, colours)):
        for i, r in enumerate(rows):
            v = r[key]
            if v is None:
                continue
            x = i + (g - .5) * (width + .02)
            ax.bar(x, 100 * v[0], width, color=colour, label=name if i == 0 else None)
            ax.errorbar(x, 100 * v[0], yerr=[[100 * (v[0] - v[1])], [100 * (v[2] - v[0])]],
                        color=INK, linewidth=1, capsize=3)
            ax.text(x, 100 * v[2] + 2, f"{100 * v[0]:.0f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(len(rows)), [f"{r['labels']} labels\n{r['recovery']}" for r in rows])
    ax.set_ylim(0, 105)
    style(ax, "Labels × recovery data (5 seeds per start)", "Success rate (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "labels_x_recovery.png")
    plt.close(fig)


def denoising() -> list[dict]:
    latency = {}
    for path in Path("outputs/latency").glob("*.json"):
        data = json.loads(path.read_text())
        for row in data["results"]:
            latency[(data["device"], row["num_steps"])] = row
    rows = []
    for steps in (1, 2, 3, 5, 10):
        policy = "lowpass_20k" if steps == 10 else f"lowpass_20k_denoise{steps}"
        all72, excluded = load(policy, "all72"), load(policy, "excluded55")
        if not all72:
            continue
        times = [r["steps_to_success"] / 20 for rs in all72.values() for r in rs
                 if r["success"]]
        rows.append({"num_steps": steps, "all72": rate(all72), "excluded": rate(excluded),
                     "force": force(all72), "time_to_insert_s": float(np.median(times)),
                     "collisions": collisions(all72),
                     "latency": {device: latency[(device, s)] for device, s in latency
                                 if s == steps}})
    return rows


def plot_denoising(rows: list[dict]) -> None:
    gpu = [r for r in rows if any("CPU" not in d for d in r["latency"])]
    if not gpu:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    for key, name, colour, marker in [("all72", "72 dataset starts", BLUE, "o"),
                                      ("excluded", "55 excluded-recording starts", ORANGE, "D")]:
        pts = [(next(v["median_ms"] for d, v in r["latency"].items() if "CPU" not in d),
                r[key], r["num_steps"]) for r in gpu if r[key] is not None]
        x = [p[0] for p in pts]
        y = np.array([p[1][:3] for p in pts]) * 100
        ax.fill_between(x, y[:, 1], y[:, 2], color=colour, alpha=.12, linewidth=0)
        ax.plot(x, y[:, 0], color=colour, linewidth=2, marker=marker, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=name)
        if key == "all72":
            for xi, yi, s in zip(x, y[:, 0], [p[2] for p in pts]):
                ax.annotate(f"{s} step{'s' if s > 1 else ''}", (xi, yi), xytext=(0, 9),
                            textcoords="offset points", ha="center", color=MUTED, fontsize=8.5)
    ax.set_xlabel("Median latency per 50-action chunk on the GPU (ms)", color=MUTED)
    ax.set_ylim(0, 100)
    style(ax, "Denoising steps: success vs inference latency", "Success rate (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "denoising_latency.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    holdout = set(split_indices(DATASET)["holdout"])

    rows = scaling(holdout)
    plot_scaling(rows)
    print("### Data scaling\n")
    print("| Demos | Policy | Trained starts | Unseen dataset starts | Excluded starts (55) "
          "| Deep-collision rollouts (55) | Median peak force (55) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        unseen_n = "" if r["unseen_dataset"] is None else f" (n={r['unseen_dataset'][3]})"
        print(f"| {r['demos']}{' (64 + mirrored)' if r['mirror'] else ''} | `{r['policy']}` | "
              f"{pct(r['trained'])} | {pct(r['unseen_dataset'])}{unseen_n} | {pct(r['excluded'])} | "
              + ("–" if r["excluded_collisions"] is None else
                 f"{100 * r['excluded_collisions']:.0f}%") + " | "
              + ("–" if r["excluded_force"] is None else f"{r['excluded_force']:.0f} N") + " |")

    rows = two_by_two(holdout)
    plot_two_by_two(rows)
    print("\n### Labels × recovery\n")
    print("| Labels | Recovery data | Policy | Training starts (64) | Holdout (8) | "
          "Excluded starts (55) | Deep-collision rollouts (64 / 55) | Median peak force (64 / 55) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['labels']} | {r['recovery']} | `{r['policy']}` | {pct(r['train64'])} | "
              f"{pct(r['holdout8'])} | {pct(r['excluded'])} | "
              f"{100 * r['collisions'][0]:.0f}% / {100 * r['collisions'][1]:.0f}% | "
              f"{r['force'][0]:.0f} / {r['force'][1]:.0f} N |")

    rows = denoising()
    plot_denoising(rows)
    devices = sorted({d for r in rows for d in r["latency"]})
    print("\n### Denoising steps and latency (low-pass 20k)\n")
    print("| Denoising steps | Seeds | 72 dataset starts | 55 excluded starts | "
          + "Median time to insert | Deep-collision rollouts (72) | "
          + " | ".join(f"Chunk latency, {d}" for d in devices) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(devices))
    for r in rows:
        lat = " | ".join(f"{r['latency'][d]['median_ms']:.0f} ms" if d in r["latency"] else "–"
                         for d in devices)
        print(f"| {r['num_steps']} | {r['all72'][4]} | {pct(r['all72'])} | {pct(r['excluded'])} | "
              f"{r['time_to_insert_s']:.2f} s | {100 * r['collisions']:.0f}% | {lat} |")


if __name__ == "__main__":
    main()

"""
viz.py
------
Renders figures + summary tables from run_benchmark.py CSVs
(sandbox/results/benchmark_{crop}.csv).

Panels per crop (wheat + maize side by side):
  1. Box plot of per-cell R2 per model        -> boxplot_r2.png
  2. ECDF of per-cell R2 per model            -> ecdf_r2.png
  3. R2 vs n_train-years scatter by model     -> scatter_ntrain.png
  4. R2 vs per-cell train-mean yield scatter  -> scatter_trainmean.png
Baselines (per_cell_train_mean, crop_mean) are shown as reference where useful.

Usage
-----
    python3 viz.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["baseline_crop_mean", "baseline_per_cell_mean",
               "01_constant", "02_co2_log", "03_climate_linear",
               "04_saturating", "05_weather_only"]

R2_CLIP = (-2.5, 1.0)      # display clip for the hugely-negative crop_mean baseline
CROP_PALS = {"wheat": "#2c7fb8", "maize": "#d95f0e"}


def load_crop_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER,
                                 ordered=True)
    return df.sort_values("model")


def order_that_exist(df: pd.DataFrame) -> list[str]:
    return [m for m in MODEL_ORDER if m in set(df["model"])]


def plot_boxplot(ax, df: pd.DataFrame, clip=R2_CLIP):
    models = order_that_exist(df)
    data = []
    n_out = []
    for m in models:
        vals = df.loc[df["model"] == m, "r2"].dropna()
        data.append(vals.clip(*clip))
        n_out.append(int(((vals < clip[0]) | (vals > clip[1])).sum()))
    bp = ax.boxplot(data, tick_labels=[m.replace("baseline_", "base_") for m in models],
                    showmeans=True, patch_artist=True, widths=0.6)
    for m, patch in zip(models, bp["boxes"]):
        patch.set_facecolor(CROP_PALS.get(m, "#cccccc"))
        patch.set_alpha(0.7)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("per-cell R² (8 test years)")
    for i, n in enumerate(n_out, start=1):
        if n:
            ax.annotate(f"{n} clipped", (i, clip[0] + 0.03),
                        ha="center", fontsize=7, color="#666666")
    ax.set_ylim(*clip)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_ecdf(ax, df: pd.DataFrame, clip=(-2.5, 1.0)):
    for m in order_that_exist(df):
        vals = df.loc[df["model"] == m, "r2"].dropna().sort_values().values
        vals = vals[(vals >= clip[0]) & (vals <= clip[1])]
        xs = np.sort(np.concatenate([vals, [clip[0], clip[1]]]))
        ys = np.linspace(0, 1, len(vals) + 2)
        ax.plot(xs, ys, lw=1.2, label=m if not m.startswith("baseline_") else f"base {m[9:]}")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlim(*clip)
    ax.set_xlabel("per-cell R²")
    ax.set_ylabel("fraction of locations")
    ax.set_title("ECDF of per-cell R²", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)


def plot_scatter_grid(df: pd.DataFrame, ycol: str, xlabel: str, fname: str,
                      ylim=R2_CLIP, figs_dir: Path | None = None, dpi: int = 140):
    models = [m for m in order_that_exist(df) if not m.startswith("baseline_")]
    if not models:
        return
    crops = sorted(df["crop"].unique())
    fig, axes = plt.subplots(len(models), len(crops),
                             figsize=(4.6 * len(crops), 2.4 * len(models)),
                             squeeze=False, sharey=False)
    for i, m in enumerate(models):
        for j, crop in enumerate(crops):
            ax = axes[i][j]
            sub = df[(df["model"] == m) & (df["crop"] == crop)]
            ax.scatter(sub[ycol], sub["r2"], s=6, alpha=0.3,
                       color=CROP_PALS[crop])
            ax.axhline(0, color="grey", lw=0.7, ls="--")
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(m.split("_", 1)[-1] if m.startswith("0") else m, fontsize=9)
            if i == len(models) - 1:
                ax.set_xlabel(xlabel)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(crop.capitalize(), fontsize=9)
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figs_dir / fname, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--figs-dir", type=Path, default=RESULTS_DIR / "figs")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for crop in ("wheat", "maize"):
        path = args.results_dir / f"benchmark_{crop}.csv"
        if path.exists():
            frames[crop] = load_crop_df(path)
    if not frames:
        print("no benchmark_*.csv found - run run_benchmark.py first")
        return
    df = pd.concat(frames.values(), ignore_index=True)
    if "crop" not in df.columns:
        df["crop"] = df["model"].map(lambda m: "wheat")

    # ---- panel 1: boxplot ----
    n_crops = len(frames)
    fig, axes = plt.subplots(1, n_crops, figsize=(5.6 * n_crops, 4.2), squeeze=False)
    for ax, crop in zip(axes[0], frames):
        plot_boxplot(ax, frames[crop])
        ax.set_title(f"{crop.title()} - per-cell R² by model", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.figs_dir / "boxplot_r2.png", dpi=args.dpi)
    plt.close(fig)

    # ---- panel 2: ECDF ----
    fig, axes = plt.subplots(1, n_crops, figsize=(5.6 * n_crops, 4.0), squeeze=False)
    for ax, crop in zip(axes[0], frames):
        plot_ecdf(ax, frames[crop])
        ax.set_title(f"{crop.title()}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.figs_dir / "ecdf_r2.png", dpi=args.dpi)
    plt.close(fig)

    # ---- panels 3 & 4: scatter grids ----
    if "n_train" in df.columns:
        plot_scatter_grid(df, "n_train", "per-location train years (381-411)",
                          "scatter_ntrain.png", figs_dir=args.figs_dir, dpi=args.dpi)
    if "train_mean" in df.columns:
        plot_scatter_grid(df, "train_mean", "per-location train-mean yield",
                          "scatter_trainmean.png", figs_dir=args.figs_dir, dpi=args.dpi)

    print(f"wrote figures to {args.figs_dir}")
    plt.close("all")


if __name__ == "__main__":
    main()
"""
evaluate_programs.py
--------------------
Competition-aligned evaluator for the FutureCrop problem.

The engine calls evaluate(prog, output_dir) with a ProgramStrings (which is
turned into `model(x, params)` and `estimate_params(x, y)`) and receives a
scalar score and a metrics dict.

Interface (see load_data.py):
    x = {"climate": (T, 240, 5), "meta": (T, 21)}   dict of float32 arrays
    y = (T,) simulated yield
    temporal split: train = years 381-409, val = years 410-419.
    Candiate functions act on x['climate'] (240-day series of tasmax/tasmin/
    pr/rsds/cumulative-rsds) and x['meta'] (fixed columns, see prep.json).

Two-phase evaluation (mirrors the competition's temporal distribution shift):
    1. FIT   : estimate_params(x_train, y_train) -> p0, then scipy L-BFGS-B
               minimises MSE on a SUBSAMPLED train set (fast, few params).
    2. EVAL  : score model(x_val, params_opt) against y_val.

Score (lower is better):
    primary = 100 * (1 - median_per_cell_R2_val)      # baseline per-location
                                                      # mean skores ~100
    then small penalties on n_params and AST depth. 999.0 on any failure.

Metrics also record global R2, per-cell R2 mean, per-cell R2 distribution and
the competition-style region production R2 (best-approximation: weighted by
the largest cells by... see notes in prompt_context.txt).
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "FunSearch"))
from FunSearch.program_parser import ProgramStrings, strings_to_callables

# --- Score weights (keep tiny: R2 term dominates) ---
PARAM_COUNT_WEIGHT = 0.01
AST_DEPTH_WEIGHT = 0.001
R2_TERM_SCALE = 100.0

# --- Fit-time settings ---
MIN_CELL_YEARS = 5        # need >= this many val years per cell to score R2
FIT_MAX_ITER = 100
FIT_MAX_ROWS = 20_000     # stochastic subsample used only for param fitting
FIT_SEED = 0
OPTIMIZER_METHOD = "L-BFGS-B"

FAILURE_SCORE = 999.0


# ---------------------------------------------------------------------------
# Public interface (called by the engine)
# ---------------------------------------------------------------------------

def evaluate(prog: ProgramStrings, output_dir: Path | None = None) -> tuple[float, dict]:
    """Score a candidate program. See module docstring for details."""
    callables = strings_to_callables(prog)
    if callables is None:
        return FAILURE_SCORE, _failure_metrics("exec error: could not parse program")

    model_fn, param_fn = callables.model, callables.estimate_params

    try:
        x_train, y_train, x_val, y_val = load()
    except Exception as e:
        return FAILURE_SCORE, _failure_metrics(f"data load error: {e}")

    # --- Phase 1: fit on (subsampled) training data ---
    rng = np.random.default_rng(FIT_SEED)
    if len(y_train) > FIT_MAX_ROWS:
        idx = rng.choice(len(y_train), size=FIT_MAX_ROWS, replace=False)
        fit_x = {"climate": x_train["climate"][idx], "meta": x_train["meta"][idx]}
        fit_y = y_train[idx]
    else:
        fit_x, fit_y = x_train, y_train

    try:
        p0 = param_fn(fit_x, fit_y)
        p0 = np.atleast_1d(np.array(p0, dtype=float))
    except Exception as e:
        return FAILURE_SCORE, _failure_metrics(f"estimate_params error: {e}")
    if not np.all(np.isfinite(p0)):
        return FAILURE_SCORE, _failure_metrics("non-finite initial params")

    try:
        def loss(params):
            pred = model_fn(fit_x, params)
            return float(np.mean((pred - fit_y) ** 2))

        result = minimize(loss, p0, method=OPTIMIZER_METHOD,
                          options={"maxiter": FIT_MAX_ITER})
        params_opt = result.x
    except Exception as e:
        return FAILURE_SCORE, _failure_metrics(f"optimizer error: {e}")

    # --- Phase 2: evaluate on validation data ---
    try:
        y_val_pred = model_fn(x_val, params_opt)
        y_val_pred = np.asarray(y_val_pred, dtype=np.float64).ravel()
        if y_val_pred.shape != y_val.shape:
            return FAILURE_SCORE, _failure_metrics(
                f"model output shape {y_val_pred.shape} != {y_val.shape}")
        if not np.all(np.isfinite(y_val_pred)):
            return FAILURE_SCORE, _failure_metrics("non-finite predictions on val set")
    except Exception as e:
        return FAILURE_SCORE, _failure_metrics(f"model eval error: {e}")

    metrics = _score_metrics(y_val, y_val_pred, x_val["meta"], params_opt, prog, result)
    _save_diagnostics(x_train, y_train, model_fn, params_opt, metrics, output_dir)

    return float(metrics["score"]), metrics


# ---------------------------------------------------------------------------
# Competition-style scoring
# ---------------------------------------------------------------------------

def _score_metrics(y_true, y_pred, meta_val, params_opt, prog, result) -> dict:
    """
    Per-cell R2 across validation years (the competition's core metric:
    the model's ability to capture weather-induced interannual variability),
    plus global R2 / MSE and a production-style weighted region R2.
    """
    y_true = np.asarray(y_true, dtype=np.float64)

    cell = meta_val[:, 0].astype(np.int64)
    cell_r2 = _per_cell_r2(y_true, y_pred, cell, min_cell_years=MIN_CELL_YEARS)

    n_ok = int(np.isfinite(cell_r2).sum())
    if n_ok == 0:
        finite_r2 = np.array([0.0])
    else:
        finite_r2 = cell_r2[np.isfinite(cell_r2)]
    median_r2 = float(np.median(finite_r2))
    mean_r2 = float(np.mean(finite_r2))

    global_r2 = _r2(y_pred, y_true)
    mse = float(np.mean((y_pred - y_true) ** 2))
    n_params = int(len(params_opt))
    depth = int(_ast_depth(prog.combined()))

    # Production-style region R2: weight cells by (harvested proxy = cell count
    # shares are unknown), so use the cell's mean predicted production proxy:
    # sqrt_average of per-cell yields is not available offline; approximate the
    # region metric with yield-weighted R2 across all val rows.
    production_r2 = _yield_weighted_r2(y_true, y_pred, cell)

    param_penalty = PARAM_COUNT_WEIGHT * n_params
    complexity_penalty = AST_DEPTH_WEIGHT * depth

    # Lower is better. Baseline (per-location mean) => median cell R2 ~= 0 => 100.
    score = R2_TERM_SCALE * (1.0 - median_r2) + param_penalty + complexity_penalty

    metrics = {
        "status": "success",
        "score": float(score),
        "median_cell_r2": median_r2,
        "mean_cell_r2": mean_r2,
        "n_cells_scored": n_ok,
        "r2_q25": float(np.percentile(finite_r2, 25)),
        "r2_q75": float(np.percentile(finite_r2, 75)),
        "production_r2": float(production_r2),
        "global_r2": float(global_r2),
        "mse_val": mse,
        "n_params": n_params,
        "ast_depth": depth,
        "param_penalty": float(param_penalty),
        "complexity_penalty": float(complexity_penalty),
        "optimizer_converged": bool(result.success),
        "params_opt": params_opt.tolist(),
        "baseline_score": float(R2_TERM_SCALE),
    }
    return metrics


def _per_cell_r2(y_true, y_pred, cell, min_cell_years: int) -> np.ndarray:
    """Per-cell R2 over years. NaN for cells with < min_cell_years val points
    or zero variance (unscorable)."""
    cells, counts = np.unique(cell, return_counts=True)
    r2 = np.full(len(cells), np.nan)

    # group reduction via cumulative sums on sorted order
    order = np.argsort(cell, kind="mergesort")
    cs = np.concatenate([[0], np.cumsum(counts)])
    yt = y_true[order]
    yp = y_pred[order]
    for i, (c, lo, hi) in enumerate(zip(cells, cs[:-1], cs[1:])):
        if hi - lo < min_cell_years:
            continue
        yc, pc = yt[lo:hi], yp[lo:hi]
        ss_tot = float(np.sum((yc - yc.mean()) ** 2))
        if ss_tot <= 1e-12:
            continue
        r2[i] = 1.0 - float(np.sum((yc - pc) ** 2)) / ss_tot
    return r2


def _yield_weighted_r2(y_true, y_pred, cell) -> float:
    """Industry-echoing aggregate: production (yield-weighted) R2 over rows.
    Harvested area is unknown locally, so weight = cell mean yield across val."""
    order = np.argsort(cell, kind="mergesort")
    cell_sorted = cell[order]
    cells, counts = np.unique(cell_sorted, return_counts=True)
    cs = np.concatenate([[0], np.cumsum(counts)])
    weights = np.zeros(len(y_true))
    yt = y_true[order]
    for i, (lo, hi) in enumerate(zip(cs[:-1], cs[1:])):
        weights[lo:hi] = max(float(yt[lo:hi].mean()), 1e-6)
    ssw = np.sum(weights * (y_true - y_pred) ** 2)
    sst = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 0.0 if sst == 0 else float(1 - ssw / sst)


def _r2(y_pred, y_true) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)


def _ast_depth(source: str) -> int:
    try:
        return _tree_depth(ast.parse(source))
    except SyntaxError:
        return 999


def _tree_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_tree_depth(c) for c in children)


def _failure_metrics(reason: str) -> dict:
    return {"status": "failed", "reason": reason, "score": FAILURE_SCORE}


# ---------------------------------------------------------------------------
# Diagnostics (train split only — the LLM never sees validation via figures)
# ---------------------------------------------------------------------------

def _save_diagnostics(x_train, y_train, model_fn, params_opt, metrics, output_dir):
    if output_dir is None:
        return
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        y_train_pred = model_fn(x_train, params_opt)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.scatter(y_train, y_train_pred, s=4, alpha=0.2, color="crimson")
        lim = [min(y_train.min(), y_train_pred.min()), max(y_train.max(), y_train_pred.max())]
        ax.plot(lim, lim, "k--", linewidth=0.8)
        ax.set_xlabel("true yield"); ax.set_ylabel("predicted yield")
        ax.set_title(f"train fit (score={metrics['score']:.2f})")

        ax = axes[1]
        cell = x_train["meta"][:, 0].astype(np.int64)
        r2 = _per_cell_r2(y_train, y_train_pred, cell, min_cell_years=MIN_CELL_YEARS)
        r2f = r2[np.isfinite(r2)]
        ax.hist(np.clip(r2f, -1, 1), bins=40, color="steelblue")
        ax.axvline(np.median(r2f), color="k", linestyle="--",
                   label=f"median {np.median(r2f):.3f}")
        ax.set_xlabel("per-cell R2 (train)"); ax.set_ylabel("cells")
        ax.set_title(f"median cell R2 (train)   metrics={len(r2f)} cells")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(output_dir / "evaluation_figure.png", dpi=120)
        plt.close(fig)
        metrics["figure"] = "evaluation_figure.png"
    except Exception as e:
        # Figure failure must never block a score result
        print(f"[evaluate] diagnostics skipped: {e}")


# ---------------------------------------------------------------------------
# Standalone smoke test — run the seed programs through the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from FunSearch.program_parser import script_to_strings

    seed_path = Path(__file__).parent / "seed_programs.py"
    programs = script_to_strings(seed_path)
    if not programs:
        print("No seed programs found — check seed_programs.py")
    else:
        for i, prog in enumerate(programs):
            print(f"\n{'='*50}\nSeed {i + 1}\n{'='*50}")
            score, metrics = evaluate(prog, output_dir=Path("/tmp/ffs_seed_test"))
            print(f"Score  : {score:.2f}")
            for k in ("median_cell_r2", "mean_cell_r2", "global_r2", "production_r2",
                      "mse_val", "n_params", "status"):
                print(f"  {k}: {metrics.get(k)}")
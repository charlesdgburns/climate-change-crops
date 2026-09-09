# AGENTS.md

## Project overview

Kaggle competition notebooks for [The Future Crop Challenge](https://www.kaggle.com/competitions/the-future-crop-challenge): predict crop yields under projected climate change from 39 years of historical data. Two approaches live in this repo:

1. **Notebooks** — deep-learning experiments (see table below). No `.py` modules, no package structure, no build system, no tests, no linting.
2. **`FreeFunSearch/`** — an equation-based (symbolic) search pipeline cloned from `https://github.com/charlesdgburns/FreeFunSearch` and wired to this competition's data. This is the active line of work.
3. **`sandbox/`** — fast, human-in-the-loop validation for scalar yield functions (dev harness; outside FreeFunSearch).

## Sandbox (validation harness)

`sandbox/` validates candidate yield-production functions quickly. Split is **train = years 381–411 (31 yrs) / test = 412–419 (8 yrs)**, derived from the prepared cache by year column. Crucially, every model is fitted **per location** (single cell): a location's own train years fit its own parameter vector, which is then scored on that same location's test years. No pooling across cells. Fits run in parallel across CPU cores.

- `sandbox/data.py` — loads the prepared caches, samples `n_cells` locations that each have ≥ `min_train_years` train-year rows, and slices per-location train/test stacks.
- `sandbox/programs/` — one script per candidate, each defining `model(...)` and `estimate_params(...)`.
- `sandbox/validate.py` — per location: `estimate_params` → scipy L-BFGS-B on `mean((pred−y)²) + ridge·Σ(params[1:]²)` (intercept exempt; `--ridge` default 0.1) → per-cell R²/MSE on that location's test years. Aggregates median/mean per-cell R², pooled MSE, pooled R² (ProcessPoolExecutor, `--n-workers`). References: `crop_mean` and `per_cell_train_mean` (both model-free).
- `sandbox/run_benchmark.py` — runs all programs over a large, fixed, seeded sample of locations (default 1000 cells/crop) and writes per-cell rows to `sandbox/results/benchmark_{crop}.csv` plus `sandbox/results/summary.md` (median/mean per-cell R², pooled MSE, % cells beating the baseline).
- `sandbox/viz.py` — renders `sandbox/results/figs/`: boxplot + ECDF of per-cell R² per model, and R² vs train-years / train-mean scatter grids.

Candidate signature (named feature args, day axis first; one location's years side by side):
`model(tasmax, tasmin, pr, rsds, cumrsds, co2, params) -> (T,)` where `tasmax…cumrsds` are `(240, T)` and `co2` is `(T,)` for that location's `T` years. `params[0]` is a per-location intercept that absorbs the location's mean yield (fit automatically, one value per location) — no `cell_mean`/`cell_id` needed. Note: `nitrogen`, `texture`, `lon`, `lat` are constant per location and thus carry no within-location signal; they are omitted from the interface.

Run: `cd sandbox && python3 validate.py --crop wheat --model all --n-cells 10`.
Benchmark + figures: `cd sandbox && python3 run_benchmark.py && python3 viz.py`.

Key analysis: `sandbox/ANALYSIS.md` — why per-cell held-out R² is negative (level-drift identity, decomposition into level-anchor + weather terms), drift forensics (trend/CO₂/recent-window all fail), weather-skill evidence (maize yes, wheat no), and the current benchmark table. Read before adding new candidate types.

## Notebooks

| Notebook | Approach |
|---|---|
| `transformer.ipynb` | Transformer with input masking (world-model style), trains on all crops |
| `transformer_simple.ipynb` | Simpler transformer variants |
| `geospatial_DNN.ipynb` | Per-location deep learning |
| `PredNet.ipynb` | PredNet-inspired architecture |
| `GraphNet` | Graph neural network (missing `.ipynb` extension — treat as notebook) |
| `submit.ipynb` | Legacy Kaggle submission (wheat `13` / maize `06`; superseded by the two explicit notebooks below) |
| `submission_baseline_13_06.ipynb` | **Kaggle submission — weather-only arm**: wheat `13`, maize `06`, per-cell closed-form ridge |
| `submission_co2_24_06.ipynb` | **Kaggle submission — literature-CO₂ arm**: wheat `24` (13-core × fixed saturating CO₂), maize `06`; only the wheat block differs from the baseline arm |

## FreeFunSearch pipeline

`FreeFunSearch/` contains the FunSearch engine (`FunSearch/`) and the crop problem definition (`problems/futurecrop/`). Twenty generations of evolution are configured via `PROBLEM_PROFILE` in `FunSearch/search_loop.py` (min = quick smoke test, full = real run).

```
FreeFunSearch/
├── FunSearch/                     # engine: search_loop, llm_caller, program_parser,
│                                  #   prompt_builder, island_manager, seed_generator
├── problems/futurecrop/
│   ├── code/
│   │   ├── prepare_data.py        # parquet -> per-location arrays + cache (run first)
│   │   ├── load_data.py           # fast cache loader (candidates import this data)
│   │   ├── evaluate_programs.py   # 2-phase scoring (scipy fit on train, R2 on val)
│   │   ├── seed_programs.py       # baseline + mechanistic starting programs
│   │   ├── prompt_context.txt     # problem description for the LLM
│   │   └── code_guide.txt         # code-generation rules for the LLM
│   ├── data/prepared/             # .npy caches built by prepare_data.py
│   └── funsearch/                 # run output: seed/, island_*/ programs + figures
├── data/                          # raw parquet (overrides for local runs)
└── .env                           # GOOGLE_API_KEY (required; gitignored)
```

`problems/futurecrop/code` must always be run from `problems/futurecrop/code` (paths are relative). The engine is run from the `FreeFunSearch/` root.

### How a run works

1. `python3 problems/futurecrop/code/prepare_data.py --n-cells 3000` — parses parquet into per-location arrays, caches to `problems/futurecrop/data/prepared/` (temporal split: train years 381–409, val 410–419; per-crop subsample).
2. `python -m FunSearch.search_loop` — seed stage (evaluates `seed_programs.py`), island initialisation and evolution via Gemini (`GOOGLE_API_KEY` in `.env`). Each LLM response is parsed, evaluated (scipy L-BFGS-B fit on subsampled train, then scored), and archived. Resumable: already-scored programs are skipped.

Candidate interface (what the LLM writes): `model(x, params) -> (N,)` and `estimate_params(x, y) -> param vector`, where `x["climate"]` is `(N, 240, 5)`, `x["meta"]` is `(N, 21)`, and `x["meta"][:, 20]` is the per-location train-mean yield (baseline). Score = `100*(1 − median per-cell R² on val) + tiny penalties`, lower is better; per-location-mean baseline ≈ 116.

## Data

All data is parquet files in `data/`. Two crops: **wheat** and **maize**.

Features per crop: `pr` (precipitation), `rsds` (solar radiation), `soil_co2`, `tas`/`tasmax`/`tasmin` (temperature).

Parquet structure: first 5 columns are metadata (`year`, `lon`, `lat`, + 2 categorical), columns index `5:` (the `'0'`–`'239'` day columns) are the 240-day time series values used as model inputs. The notebook-era claim "columns 35+" is wrong — verify with `df.columns` before slicing.

Targets: `train_solutions_wheat.parquet` and `train_solutions_maize.parquet`.

### Data path convention

Notebooks were written for Kaggle (`/kaggle/input/the-future-crop-challenge/`). When running locally, use the `data/` directory instead. The `DATA_DIR` variable in each notebook must be pointed at `data/`. The FreeFunSearch pipeline reads `data/` via `DEFAULT_DATA_DIR` in `prepare_data.py`. Train years are 381–419; test years are 420–497 (CO₂ rises to ~1108 ppm in test — see `sandbox/CO2_LIT.md`; the "~560 ppm" claim is wrong).

"""Stage 3: iterative hyperparameter tuning on the VALIDATION week (Dec 9-15).

Every run is appended to results/experiments.csv, so the full tuning history is
documented. Change one or two things per experiment and write down why.

Examples (PowerShell, from the repo root):
  python -m scripts.03_tune --model persistence --note "reference"
  python -m scripts.03_tune --model lstm --note "defaults"
  python -m scripts.03_tune --model lstm --set window=288 --note "ACF: test 2-day window"
  python -m scripts.03_tune --model sarimax --grid daily_k=4,8,12 --grid weekly_k=2,4 --note "stage 1"
"""
import argparse
import itertools
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.features import compute_metrics, split_positions
from src.processed import load_matrix, load_top_squares, square_series
from src.runner import DEFAULTS, run

LOG = C.RESULTS_DIR / "experiments.csv"


def parse_value(v):
    try:
        return json.loads(v.lower() if v.lower() in ("true", "false") else v)
    except json.JSONDecodeError:
        return v


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DEFAULTS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2,...")
    ap.add_argument("--square", type=int, default=None, help="default: top-1 square")
    ap.add_argument("--seed", type=int, default=C.SEED, help="random seed (neural models)")
    ap.add_argument("--note", default="")
    return ap.parse_args()


def combos(args):
    fixed = {k: parse_value(v) for k, v in (s.split("=", 1) for s in args.set)}
    grid = {k: [parse_value(x) for x in v.split(",")] for k, v in (g.split("=", 1) for g in args.grid)}
    for unknown in set(fixed) | set(grid):
        if unknown not in DEFAULTS[args.model]:
            raise SystemExit(f"Unknown parameter '{unknown}' for {args.model}. "
                             f"Valid: {list(DEFAULTS[args.model])}")
    keys = list(grid)
    for values in itertools.product(*grid.values()) if keys else [()]:
        yield {**fixed, **dict(zip(keys, values))}


def plot_curve(history, exp_id, model):
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(h.epoch, h.train_mse, label="train")
    ax.plot(h.epoch, h.val_mse, label="validation")
    ax.set(xlabel="Epoch", ylabel="MSE (scaled log)", title=f"Exp {exp_id}: {model}", yscale="log")
    ax.legend(frameon=False)
    out = C.FIG_DIR / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"exp{exp_id:03d}_{model}_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    M, index = load_matrix()
    square = args.square or load_top_squares()[0]
    y = square_series(M, index, square).to_numpy()
    sp = split_positions(index)
    actual = y[sp["val_start"]:sp["val_end"]]
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()

    for params in combos(args):
        exp_id = int(log["exp_id"].max()) + 1 if len(log) else 1
        print(f"\n=== Experiment {exp_id}: {args.model} {params or '(defaults)'} on square {square}")
        predict_fn, info = run(args.model, params, y, index, sp, sp["val_start"], sp["val_end"],
                              seed=args.seed)
        pred = predict_fn()
        m = compute_metrics(actual, pred)
        row = {"exp_id": exp_id, "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "model": args.model, "square": square, "seed": args.seed, "changed": json.dumps(params),
               "val_MAE": m["MAE"], "val_RMSE": m["RMSE"], "val_MAPE_%": m["MAPE_%"],
               "train_s": info.get("train_seconds"), "epochs_run": info.get("epochs_run"),
               "best_epoch": info.get("best_epoch"), "n_params": info.get("n_params"),
               "aic": info.get("aic"), "note": args.note,
               "all_params": json.dumps(info["params"])}
        if "history" in info:
            plot_curve(info["history"], exp_id, args.model)
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        log.to_csv(LOG, index=False)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
               if k not in ("all_params", "time")})

    same = log[(log.model == args.model) & (log.square == square)].sort_values("val_MAE")
    print(f"\n--- Leaderboard: {args.model}, square {square} (validation week) ---")
    print(same[["exp_id", "changed", "seed", "val_MAE", "val_RMSE", "val_MAPE_%", "train_s",
                "best_epoch", "note"]].round(3).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

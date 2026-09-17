"""Stage 4: final evaluation on the TEST week (Dec 16-22) for the top-3 squares.

Uses the tuned hyperparameters in configs/final_params.json. Run once, after tuning.
  python -m scripts.04_final

Tuning (results/experiments.csv) showed that the random seed changes neural-network
validation MAE by up to ~30, more than any hyperparameter. Therefore LSTM and TCN are
trained with 3 seeds per area and reported as mean +/- std; plots show the median-seed
run. SARIMAX and the reference models are deterministic (one run).

Outputs (results/final/, figures/final/):
  metrics_square_<id>.csv   one table per area (MAE, RMSE, MAPE; mean and std over seeds)
  summary_all_squares.csv   mean MAE / RMSE / MAPE of every model in every area
  runs_all.csv              every single run (square x model x seed)
  timing.csv, timing_summary.csv, timing_method.txt, hardware.json
  predictions_<id>.csv      actual + forecasts (all seeds) for failure analysis
  daily_mae_<id>.csv
  fig_<id>_<model>.png      9 actual-vs-predicted plots
  fig_error_by_day.png, fig_error_by_hour.png, fig_worst_day_<id>.png
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.features import compute_metrics, split_positions
from src.processed import load_matrix, load_top_squares, square_series
from src.runner import MAIN_MODELS, REFERENCE_MODELS, hardware_info, run

SEEDS = [42, 1, 2]
NEURAL = ["lstm", "tcn"]
ALL_MODELS = REFERENCE_MODELS + MAIN_MODELS
RES = C.RESULTS_DIR / "final"
FIG = C.FIG_DIR / "final"
NAMES = {"sarimax": "SARIMAX-Fourier", "lstm": "LSTM", "tcn": "TCN",
         "persistence": "Persistence", "seasonal_naive": "Seasonal naive (1 week)"}
COLORS = {"sarimax": "#2e8b57", "lstm": "#d1495b", "tcn": "#edae49"}
METRICS = ["MAE", "RMSE", "MAPE_%"]


def fmt_time_axis(ax, fmt="%a %d"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=C.TZ))


def plot_forecast(ts, actual, pred, square, model, subtitle):
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True, height_ratios=[3, 1])
    ax[0].plot(ts, actual, color="#222222", lw=.9, label="Actual")
    ax[0].plot(ts, pred, color=COLORS[model], lw=.9, alpha=.85,
               label=f"{NAMES[model]} (one-step-ahead)")
    ax[0].set(ylabel="Internet activity", title=f"Square {square} — {NAMES[model]} — test week Dec 16–22\n{subtitle}")
    ax[0].legend(frameon=False, loc="upper left")
    ax[1].plot(ts, pred - actual, color="#555555", lw=.6)
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_ylabel("Forecast − actual")
    ax[1].xaxis.set_major_locator(mdates.DayLocator(tz=C.TZ))
    fmt_time_axis(ax[1])
    fig.savefig(FIG / f"fig_{square}_{model}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with open(C.ROOT / "configs" / "final_params.json") as f:
        final_params = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    M, index = load_matrix()
    sp = split_positions(index)
    s0, s1 = sp["test_start"], sp["test_end"]
    ts = index[s0:s1]
    squares = load_top_squares()
    hw = hardware_info()
    with open(RES / "hardware.json", "w") as f:
        json.dump(hw, f, indent=2)
    print("Hardware:", hw)
    print("Final parameters:", final_params)

    runs, daily_all, hourly_top = [], [], None
    for square in squares:
        y = square_series(M, index, square).to_numpy()
        actual = y[s0:s1]
        preds = pd.DataFrame({"timestamp": ts.astype(str), "actual": actual})
        table = []

        for model in ALL_MODELS:
            seeds = SEEDS if model in NEURAL else [C.SEED]
            model_runs = []
            for seed in seeds:
                print(f"\nSquare {square} | {NAMES[model]} | seed {seed}")
                predict_fn, info = run(model, final_params.get(model, {}), y, index, sp, s0, s1, seed=seed)
                times = []
                for _ in range(C.N_TIMING_REPEATS):
                    t0 = time.perf_counter()
                    pred = predict_fn()
                    times.append(time.perf_counter() - t0)
                m = compute_metrics(actual, pred)
                print({k: round(v, 3) for k, v in m.items()}, f"train {info.get('train_seconds', 0):.1f}s")
                row = {"square": square, "model": NAMES[model], "seed": seed if model in NEURAL else None, **m,
                       "train_seconds": info.get("train_seconds", 0.0),
                       "epochs_run": info.get("epochs_run"), "best_epoch": info.get("best_epoch"),
                       "seconds_per_epoch": info.get("seconds_per_epoch"), "n_params": info.get("n_params"),
                       "inference_week_mean_s": float(np.mean(times)),
                       "inference_week_std_s": float(np.std(times)),
                       "inference_ms_per_step": float(np.mean(times)) / len(actual) * 1000}
                runs.append(row)
                model_runs.append((m["MAE"], seed, pred, m))
                if model in NEURAL:
                    preds[f"{model}_seed{seed}"] = pred

            maes = np.array([r[0] for r in model_runs])
            rep = sorted(model_runs, key=lambda r: r[0])[len(model_runs) // 2]   # median-MAE run
            preds[model] = rep[2]
            entry = {"model": NAMES[model], "type": "reference" if model in REFERENCE_MODELS else "main",
                     "n_runs": len(model_runs)}
            for k in METRICS:
                vals = np.array([r[3][k] for r in model_runs])
                entry[f"{k}_mean"] = vals.mean()
                entry[f"{k}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            table.append(entry)

            if model in MAIN_MODELS:
                if model in NEURAL:
                    sub = (f"Plotted: seed {rep[1]} (median of {len(maes)}), MAE {rep[0]:.1f}   |   "
                           f"over seeds: MAE {entry['MAE_mean']:.1f} ± {entry['MAE_std']:.1f}, "
                           f"RMSE {entry['RMSE_mean']:.1f}, MAPE {entry['MAPE_%_mean']:.2f}%")
                else:
                    sub = f"MAE {rep[3]['MAE']:.1f} | RMSE {rep[3]['RMSE']:.1f} | MAPE {rep[3]['MAPE_%']:.2f}%"
                plot_forecast(ts, actual, rep[2], square, model, sub)

        tdf = pd.DataFrame(table)
        tdf.round(3).to_csv(RES / f"metrics_square_{square}.csv", index=False)
        print(f"\n=== Square {square}: test-week metrics ===")
        print(tdf.round(2).to_string(index=False))
        preds.to_csv(RES / f"predictions_{square}.csv", index=False)

        # errors by day / hour (median-seed predictions)
        err = preds[["actual"] + ALL_MODELS].copy()
        for mdl in ALL_MODELS:
            err[mdl] = (err[mdl] - err["actual"]).abs()
        err["day"] = ts.strftime("%a %d")
        daily = err.groupby("day", sort=False)[ALL_MODELS].mean()
        daily.round(2).to_csv(RES / f"daily_mae_{square}.csv")
        daily_all.append((square, daily))

        if square == squares[0]:
            ape = err[MAIN_MODELS].div(preds["actual"], axis=0) * 100
            hourly_top = ape.groupby(ts.hour).mean()
            # worst day for the main models: actual vs all three forecasts
            worst = daily[MAIN_MODELS].mean(axis=1).idxmax()
            mask = (err["day"] == worst).to_numpy()
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(ts[mask], preds["actual"][mask], color="#222222", lw=1.6, label="Actual")
            for mdl in MAIN_MODELS:
                ax.plot(ts[mask], preds[mdl][mask], color=COLORS[mdl], lw=1, label=NAMES[mdl])
            ax.set(title=f"Square {square}: worst test day for the main models ({worst})",
                   ylabel="Internet activity")
            fmt_time_axis(ax, "%H:%M")
            ax.legend(frameon=False)
            fig.savefig(FIG / f"fig_worst_day_{square}.png", bbox_inches="tight", dpi=170)
            plt.close(fig)

    # ---------------- summaries
    rdf = pd.DataFrame(runs)
    rdf.round(4).to_csv(RES / "runs_all.csv", index=False)
    summary = (rdf.groupby(["model", "square"], sort=False)[METRICS].mean()
               .unstack("square").round(2))
    summary.to_csv(RES / "summary_all_squares.csv")
    print("\n=== Mean test metrics per area (neural: mean over 3 seeds) ===\n", summary.to_string())

    timing = rdf.groupby("model", sort=False).agg(
        runs=("train_seconds", "size"),
        train_mean_s=("train_seconds", "mean"), train_std_s=("train_seconds", "std"),
        epochs_mean=("epochs_run", "mean"), seconds_per_epoch=("seconds_per_epoch", "mean"),
        inference_week_mean_s=("inference_week_mean_s", "mean"),
        inference_ms_per_step=("inference_ms_per_step", "mean"), n_params=("n_params", "mean"))
    rdf[["square", "model", "seed", "train_seconds", "epochs_run", "best_epoch", "seconds_per_epoch",
         "inference_week_mean_s", "inference_week_std_s", "inference_ms_per_step",
         "n_params"]].round(4).to_csv(RES / "timing.csv", index=False)
    timing.round(4).to_csv(RES / "timing_summary.csv")
    print("\n=== Timing (all runs: 3 areas x seeds) ===\n", timing.round(3).to_string())
    (RES / "timing_method.txt").write_text(
        "Wall-clock times measured with time.perf_counter() in a single Python process, CPU only.\n"
        "Training time: fitting on the training split only (Nov 1 - Dec 8). For LSTM/TCN this includes every "
        "epoch until early stopping on the validation week (Dec 9 - 15). SARIMAX: one fit per area. "
        f"LSTM/TCN: one training per seed {SEEDS} per area (9 runs per model). Reported as mean and standard "
        "deviation over all runs of a model.\n"
        f"Inference time: producing all {s1 - s0} one-step-ahead forecasts of the test week, repeated "
        f"{C.N_TIMING_REPEATS} times per run after a warm-up call; mean reported (per week and per step).\n"
        "Other applications were closed during the measurements.\n"
        f"Hardware: {json.dumps(hw)}\n")

    # ---------------- failure-analysis figures
    fig, axes = plt.subplots(1, len(daily_all), figsize=(5.4 * len(daily_all), 3.6))
    for ax, (square, daily) in zip(np.atleast_1d(axes), daily_all):
        im = ax.imshow(daily[MAIN_MODELS].T.to_numpy(), aspect="auto", cmap="Reds")
        ax.set(xticks=range(len(daily)), yticks=range(len(MAIN_MODELS)), title=f"Square {square}: MAE per day")
        ax.set_xticklabels(daily.index, rotation=45, ha="right")
        ax.set_yticklabels([NAMES[m] for m in MAIN_MODELS])
        fig.colorbar(im, ax=ax)
    fig.savefig(FIG / "fig_error_by_day.png", bbox_inches="tight", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for mdl in MAIN_MODELS:
        ax.plot(hourly_top.index, hourly_top[mdl], marker="o", ms=3, color=COLORS[mdl], label=NAMES[mdl])
    ax.set(xlabel="Hour of day", ylabel="Mean absolute % error", xticks=range(0, 24, 2),
           title=f"Square {squares[0]}: error by hour of day (test week)")
    ax.legend(frameon=False)
    fig.savefig(FIG / "fig_error_by_hour.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("\nDone. See results/final and figures/final")


if __name__ == "__main__":
    main()

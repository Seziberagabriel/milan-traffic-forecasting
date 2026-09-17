"""Stage 5: diagnostics for the Results / Failure-analysis sections.

Run after scripts/04_final.py:
  python -m scripts.05_diagnostics                    # analysis only (~1 min)
  python -m scripts.05_diagnostics --ablation lstm    # + post-hoc test: LSTM without weekly lag (~3 min)
  python -m scripts.05_diagnostics --ablation lstm tcn  # + TCN too (~20 min)

The ablation is a POST-HOC DIAGNOSTIC on the test week, run after model selection was
frozen. It explains a failure; it must not be used to choose the final model.

Outputs: results/final/diagnostics.json, results/final/ablation_weekly_lag.csv,
         figures/final/fig_week_to_week_change.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.features import compute_metrics, split_positions
from src.processed import load_matrix, load_top_squares, square_series
from src.runner import run

RES = C.RESULTS_DIR / "final"
FIG = C.FIG_DIR / "final"
MODELS = ["persistence", "seasonal_naive", "sarimax", "lstm", "tcn"]
SEEDS = [42, 1, 2]


def error_profile(df):
    """Where do the errors come from? (median-seed predictions from 04_final)"""
    actual = df["actual"].to_numpy()
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)).tz_convert(C.TZ)
    step_change = np.abs(np.diff(actual, prepend=actual[0]))
    ramps = step_change >= np.quantile(step_change, 0.9)      # 10% largest 10-min changes
    daytime = (ts.hour >= 8) & (ts.hour < 22)
    weekend = ts.dayofweek >= 5
    out = {}
    for m in MODELS:
        p = df[m].to_numpy()
        err = p - actual
        ae = np.abs(err)
        k = len(ae) // 10
        out[m] = {
            "bias_mean_error": float(err.mean()),
            "bias_pct_of_mean_traffic": float(err.mean() / actual.mean() * 100),
            "corr_pred_t_vs_actual_t": float(np.corrcoef(p, actual)[0, 1]),
            "corr_pred_t_vs_actual_t_minus_1": float(np.corrcoef(p[1:], actual[:-1])[0, 1]),
            "MAE_on_ramps_top10pct_changes": float(ae[ramps].mean()),
            "MAE_other_steps": float(ae[~ramps].mean()),
            "MAE_daytime_08_22": float(ae[daytime].mean()),
            "MAE_night": float(ae[~daytime].mean()),
            "MAE_weekdays": float(ae[~weekend].mean()),
            "MAE_weekend_Dec21_22": float(ae[weekend].mean()),
            "share_of_total_abs_error_from_worst_10pct_steps": float(np.sort(ae)[-k:].sum() / ae.sum()),
        }
    return out


def week_to_week(s, sp):
    """How different is each week from the week before? (what a weekly lag relies on)"""
    prev = s[sp["val_start"] - C.WEEK:sp["val_start"]]     # Dec 2-8
    val = s[sp["val_start"]:sp["val_end"]]                  # Dec 9-15
    test = s[sp["test_start"]:sp["test_end"]]               # Dec 16-22
    daily = lambda x: x.reshape(7, 144).sum(axis=1)
    return {
        "val_total_over_previous_week": float(val.sum() / prev.sum()),
        "test_total_over_val_week": float(test.sum() / val.sum()),
        "val_daily_ratio_vs_previous_week_MonToSun": np.round(daily(val) / daily(prev), 3).tolist(),
        "test_daily_ratio_vs_val_week_MonToSun": np.round(daily(test) / daily(val), 3).tolist(),
        "mean_abs_weekly_change_pct_val": float(np.mean(np.abs(val - prev)) / val.mean() * 100),
        "mean_abs_weekly_change_pct_test": float(np.mean(np.abs(test - val)) / test.mean() * 100),
    }


def plot_week_change(M, index, squares, sp):
    s0, s1 = sp["test_start"], sp["test_end"]
    ts = index[s0:s1]
    fig, axes = plt.subplots(len(squares), 1, figsize=(13, 2.8 * len(squares)), sharex=True)
    for ax, sq in zip(np.atleast_1d(axes), squares):
        s = square_series(M, index, sq).to_numpy()
        ax.plot(ts, s[s0 - C.WEEK:s1 - C.WEEK], color="#8aa4c8", lw=1, label="Previous week (Dec 9–15), shifted +7 days")
        ax.plot(ts, s[s0:s1], color="#222222", lw=.9, label="Test week (Dec 16–22)")
        ax.set_title(f"Square {sq}", loc="left", fontsize=10)
        ax.set_ylabel("Activity")
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=C.TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d", tz=C.TZ))
    fig.suptitle("What the weekly lag / seasonal naive sees: test week vs the week before", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_week_to_week_change.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def ablation(models, M, index, squares, sp):
    with open(C.ROOT / "configs" / "final_params.json") as f:
        params = json.load(f)
    s0, s1 = sp["test_start"], sp["test_end"]
    rows = []
    for sq in squares:
        y = square_series(M, index, sq).to_numpy()
        for m in models:
            for seed in SEEDS:
                print(f"Ablation | square {sq} | {m} without weekly lag | seed {seed}")
                fn, info = run(m, {**params[m], "weekly_lag": False}, y, index, sp, s0, s1, seed=seed)
                rows.append({"square": sq, "model": m.upper(), "seed": seed,
                             **compute_metrics(y[s0:s1], fn()), "train_seconds": info["train_seconds"]})
    abl = pd.DataFrame(rows)
    main = pd.read_csv(RES / "runs_all.csv")
    main = main[main["model"].isin([m.upper() for m in models])]
    agg = lambda d: d.groupby(["square", "model"])["MAE"].agg(["mean", "std"])
    table = agg(main).join(agg(abl), lsuffix="_with_weekly_lag", rsuffix="_without_weekly_lag")
    table["MAE_change_when_removed"] = table["mean_without_weekly_lag"] - table["mean_with_weekly_lag"]
    abl.round(3).to_csv(RES / "ablation_weekly_lag_runs.csv", index=False)
    table.round(2).to_csv(RES / "ablation_weekly_lag.csv")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", nargs="*", default=[], choices=["lstm", "tcn"])
    args = ap.parse_args()

    M, index = load_matrix()
    sp = split_positions(index)
    squares = load_top_squares()
    result = {}
    for sq in squares:
        df = pd.read_csv(RES / f"predictions_{sq}.csv")
        s = square_series(M, index, sq).to_numpy()
        result[str(sq)] = {"week_to_week": week_to_week(s, sp), "error_profile": error_profile(df)}
    plot_week_change(M, index, squares, sp)
    with open(RES / "diagnostics.json", "w") as f:
        json.dump(result, f, indent=2)

    for sq in squares:
        print(f"\n===== Square {sq} =====")
        print("Week-to-week:", json.dumps(result[str(sq)]["week_to_week"]))
        print(pd.DataFrame(result[str(sq)]["error_profile"]).T.round(3).to_string())

    if args.ablation:
        table = ablation(args.ablation, M, index, squares, sp)
        print("\n===== POST-HOC ablation on test week: weekly lag removed =====")
        print(table.round(2).to_string())
    print("\nSaved results/final/diagnostics.json and figures/final/fig_week_to_week_change.png")


if __name__ == "__main__":
    main()

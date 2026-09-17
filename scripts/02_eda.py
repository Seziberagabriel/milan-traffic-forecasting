"""Stage 2: exploratory analysis.

Run from the repo root:  python -m scripts.02_eda
Figures -> figures/ ; numbers -> results/eda_summary.json

Analyses:
  2.1 Distribution of total traffic across the 10,000 squares
  2.2 First two weeks for top-3 squares + squares 4159 and 4556
  2.3 Analysis A (top square): autocorrelation structure (ACF / PACF)
  2.4 Analysis B (top square): multi-seasonal decomposition, stationarity, anomalies
Only data up to the end of the validation period (Dec 15) is used for 2.3/2.4,
so the test week (Dec 16-22) never influences modelling decisions.
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src import config as C
from src.processed import load_grid_centroids, load_matrix, load_top_squares, square_series

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InterpolationWarning.*")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
DAY = 144    # slots per day
WEEK = 1008  # slots per week


def save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}")


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n, cum = len(x), np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


# ---------------------------------------------------------------- 2.1
def analyse_distribution(totals):
    s = pd.Series(totals)
    desc = np.sort(totals)[::-1]
    share = np.cumsum(desc) / desc.sum()
    stats = {
        "n_squares": int(len(s)),
        "squares_with_zero_total": int((s == 0).sum()),
        "mean": float(s.mean()), "median": float(s.median()),
        "mean_over_median": float(s.mean() / s.median()),
        "std": float(s.std()), "skewness": float(s.skew()), "kurtosis": float(s.kurt()),
        "p01": float(s.quantile(.01)), "p25": float(s.quantile(.25)),
        "p75": float(s.quantile(.75)), "p99": float(s.quantile(.99)), "max": float(s.max()),
        "max_over_median": float(s.max() / s.median()),
        "share_top_1pct": float(share[99]), "share_top_10pct": float(share[999]),
        "squares_for_50pct_of_traffic": int(np.searchsorted(share, 0.5) + 1),
        "gini": gini(totals),
    }

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    pos = totals[totals > 0]
    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 60)
    ax[0].hist(pos, bins=bins, color="#3b6ea5", edgecolor="white", lw=.4)
    ax[0].axvline(stats["median"], color="k", ls="--", lw=1, label="median")
    ax[0].axvline(stats["mean"], color="#d1495b", ls="--", lw=1, label="mean")
    ax[0].set(xscale="log", xlabel="Total Internet activity per square (log scale)",
              ylabel="Number of squares", title="(a) Distribution across squares")
    ax[0].legend(frameon=False)

    frac = np.arange(1, len(desc) + 1) / len(desc) * 100
    ax[1].plot(frac, share * 100, color="#3b6ea5")
    ax[1].plot([0, 100], [0, 100], color="grey", ls=":", lw=1, label="perfectly even")
    ax[1].axvline(10, color="#d1495b", ls="--", lw=1)
    ax[1].annotate(f"top 10% of squares\n= {stats['share_top_10pct']*100:.1f}% of traffic",
                   (10, stats["share_top_10pct"] * 100), xytext=(30, 45),
                   arrowprops=dict(arrowstyle="->"))
    ax[1].set(xlabel="% of squares (busiest first)", ylabel="Cumulative % of traffic",
              title=f"(b) Concentration (Gini = {stats['gini']:.2f})")
    ax[1].legend(frameon=False, loc="lower right")

    cent = load_grid_centroids()
    if cent is not None:
        v = np.log10(totals[cent["square_id"].to_numpy() - 1] + 1)
        sc = ax[2].scatter(cent["lon"], cent["lat"], c=v, s=2.2, marker="s", cmap="magma")
        ax[2].set(xlabel="Longitude", ylabel="Latitude")
    else:
        sc = ax[2].imshow(np.log10(totals.reshape(100, 100) + 1), origin="lower", cmap="magma")
        ax[2].set(xlabel="Grid column", ylabel="Grid row")
    ax[2].set_title("(c) Spatial pattern")
    fig.colorbar(sc, ax=ax[2], label="log10(total activity + 1)")
    save(fig, "fig1_total_traffic_distribution.png")
    return stats


# ---------------------------------------------------------------- 2.2
def analyse_two_weeks(M, index, squares, labels):
    start = pd.Timestamp(C.START, tz=C.TZ)
    mask = (index >= start) & (index < start + pd.Timedelta(days=14))
    days = pd.date_range(start, periods=14, freq="D")

    fig, axes = plt.subplots(len(squares), 1, figsize=(14, 2.2 * len(squares)), sharex=True)
    per_square = {}
    for ax, sq, lab in zip(axes, squares, labels):
        s = square_series(M, index, sq)
        w = s[mask]
        for d in days:
            if d.dayofweek >= 5:
                ax.axvspan(d, d + pd.Timedelta(days=1), color="grey", alpha=.12, lw=0)
        ax.plot(w.index, w.values, lw=.8, color="#3b6ea5")
        ax.set_title(f"Square {sq} — {lab}", loc="left", fontsize=10)
        ax.set_ylabel("Activity")

        wk = w[w.index.dayofweek < 5]
        we = w[w.index.dayofweek >= 5]
        prof = wk.groupby(wk.index.hour).mean()
        per_square[str(sq)] = {
            "label": lab,
            "total_full_period": float(s.sum()),
            "mean_2w": float(w.mean()), "cv_2w": float(w.std() / w.mean()) if w.mean() else None,
            "weekend_over_weekday_mean": float(we.mean() / wk.mean()) if wk.mean() else None,
            "weekday_peak_hour": int(prof.idxmax()), "weekday_min_hour": int(prof.idxmin()),
            "night_over_day_ratio": float(w.between_time("02:00", "05:00").mean() /
                                          w.between_time("10:00", "18:00").mean())
            if w.between_time("10:00", "18:00").mean() else None,
            "max_over_mean_2w": float(w.max() / w.mean()) if w.mean() else None,
            "zero_fraction_2w": float((w == 0).mean()),
        }
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(tz=C.TZ))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz=C.TZ))
    fig.suptitle("Internet activity, first two weeks (grey = weekend)", y=1.0)
    fig.tight_layout()
    save(fig, "fig2_two_weeks_five_squares.png")

    # Shape comparison: average daily profile normalised by each square's own mean
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for sq, lab in zip(squares, labels):
        w = square_series(M, index, sq)[mask]
        w = w / w.mean()
        for i, (name, part) in enumerate([("Weekdays", w[w.index.dayofweek < 5]),
                                          ("Weekends", w[w.index.dayofweek >= 5])]):
            p = part.groupby(part.index.hour + part.index.minute / 60).mean()
            ax[i].plot(p.index, p.values, lw=1.4, label=f"{sq} ({lab})")
            ax[i].set(title=f"{name}: mean daily profile (normalised)", xlabel="Hour of day",
                      xticks=range(0, 25, 3))
    ax[0].set_ylabel("Activity / square mean")
    ax[1].legend(frameon=False, fontsize=8)
    save(fig, "fig3_normalised_daily_profiles.png")

    cent = load_grid_centroids()
    if cent is not None:
        c = cent.set_index("square_id")
        for sq in squares:
            per_square[str(sq)]["centroid_lat_lon"] = [round(c.loc[sq, "lat"], 5),
                                                        round(c.loc[sq, "lon"], 5)]
    return per_square


# ---------------------------------------------------------------- 2.3
def analysis_autocorrelation(y):
    n = len(y)
    ac = acf(y, nlags=2 * WEEK, fft=True)
    pa = pacf(y, nlags=48, method="ywm")
    yd = (y - y.shift(DAY)).dropna()
    acd = acf(yd, nlags=2 * WEEK, fft=True)
    conf = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(3, 1, figsize=(14, 9))
    hrs = np.arange(len(ac)) / 6
    ax[0].plot(hrs, ac, lw=.8, color="#3b6ea5")
    ax[0].fill_between(hrs, -conf, conf, color="grey", alpha=.25)
    for h in range(24, 24 * 14 + 1, 24):
        ax[0].axvline(h, color="#d1495b" if h % 168 == 0 else "grey",
                      lw=1 if h % 168 == 0 else .4, ls=":")
    ax[0].set(title="(a) ACF of raw series, lags up to 2 weeks (red = weekly lag)",
              xlabel="Lag (hours)", ylabel="ACF")
    ax[1].bar(np.arange(len(pa)), pa, color="#3b6ea5", width=.6)
    ax[1].axhspan(-conf, conf, color="grey", alpha=.25)
    ax[1].set(title="(b) PACF, first 48 lags (8 hours)", xlabel="Lag (10-min steps)", ylabel="PACF")
    ax[2].plot(hrs, acd, lw=.8, color="#2e8b57")
    ax[2].fill_between(hrs, -conf, conf, color="grey", alpha=.25)
    for h in range(168, 24 * 14 + 1, 168):
        ax[2].axvline(h, color="#d1495b", ls=":")
    ax[2].set(title="(c) ACF after daily seasonal differencing (y_t − y_{t−144})",
              xlabel="Lag (hours)", ylabel="ACF")
    fig.tight_layout()
    save(fig, "fig4_top_square_acf_pacf.png")

    return {
        "n_obs": n,
        "acf_lag1_10min": float(ac[1]), "acf_lag6_1h": float(ac[6]),
        "acf_lag36_6h": float(ac[36]), "acf_lag72_12h": float(ac[72]),
        "acf_lag144_1day": float(ac[DAY]), "acf_lag1008_1week": float(ac[WEEK]),
        "pacf_first_10": [round(float(v), 3) for v in pa[1:11]],
        "pacf_significant_lags_upto48": [int(i) for i in np.where(np.abs(pa[1:]) > conf)[0] + 1],
        "diff144_acf_lag1": float(acd[1]), "diff144_acf_lag144": float(acd[DAY]),
        "diff144_acf_lag1008": float(acd[WEEK]),
    }


# ---------------------------------------------------------------- 2.4
def stationarity_tests(x):
    x = x.dropna()
    adf = adfuller(x, autolag="AIC")
    k = kpss(x, regression="c", nlags="auto")
    return {"adf_stat": float(adf[0]), "adf_p": float(adf[1]), "adf_lags": int(adf[2]),
            "kpss_stat": float(k[0]), "kpss_p": float(k[1])}


def analysis_decomposition(y):
    res = MSTL(y, periods=(DAY, WEEK), stl_kwargs={"robust": True}).fit()
    S_d, S_w = res.seasonal.iloc[:, 0], res.seasonal.iloc[:, 1]
    T, R = res.trend, res.resid

    def strength(S):
        return float(max(0.0, 1 - R.var() / (S + R).var()))

    z = (R - R.median()) / (1.4826 * (R - R.median()).abs().median())
    anomalies = z[z.abs() > 4]
    top_anom = anomalies.abs().sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
    ax[0].plot(y.index, y, lw=.6, color="#3b6ea5")
    ax[0].scatter(anomalies.index, y.loc[anomalies.index], s=10, color="#d1495b",
                  zorder=3, label="|robust z| > 4")
    ax[0].legend(frameon=False, loc="upper left")
    for a, series, name in zip(ax, [y, T, S_d, S_w, R],
                               ["Observed", "Trend", "Daily seasonal (144)",
                                "Weekly seasonal (1008)", "Residual"]):
        if name != "Observed":
            a.plot(series.index, series, lw=.6, color="#333333")
        a.set_ylabel(name, fontsize=9)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=C.TZ))
    fig.suptitle("MSTL decomposition of the top square (Nov 1 – Dec 15)", y=1.0)
    fig.tight_layout()
    save(fig, "fig5_top_square_mstl.png")

    return {
        "seasonal_strength_daily": strength(S_d),
        "seasonal_strength_weekly": strength(S_w),
        "trend_strength": float(max(0.0, 1 - R.var() / (T + R).var())),
        "variance_share": {k: float(v.var() / y.var()) for k, v in
                           {"trend": T, "daily": S_d, "weekly": S_w, "resid": R}.items()},
        "stationarity_raw": stationarity_tests(y),
        "stationarity_diff144": stationarity_tests(y - y.shift(DAY)),
        "stationarity_diff1_diff144": stationarity_tests((y - y.shift(DAY)).diff()),
        "n_anomalies_absz_gt4": int(len(anomalies)),
        "top_anomalies": [{"time": str(t), "robust_z": round(float(z[t]), 2),
                           "observed": round(float(y[t]), 1)} for t in top_anom.index],
    }


def main():
    M, index = load_matrix()
    top3 = load_top_squares()
    squares = top3 + C.SPECIAL_SQUARES
    labels = ["Top 1", "Top 2", "Top 3", "Special", "Special"]
    summary = {}

    print("2.1 distribution")
    totals = np.asarray(M.sum(axis=0), dtype=np.float64)
    summary["distribution"] = analyse_distribution(totals)

    print("2.2 two-week comparison")
    summary["five_squares"] = analyse_two_weeks(M, index, squares, labels)

    y = square_series(M, index, top3[0]).loc[:pd.Timestamp(C.VAL_END, tz=C.TZ)]
    print("2.3 autocorrelation")
    summary["analysis_A_autocorrelation"] = analysis_autocorrelation(y)
    print("2.4 decomposition + stationarity + anomalies (may take ~1 min)")
    summary["analysis_B_decomposition"] = analysis_decomposition(y)

    with open(C.RESULTS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

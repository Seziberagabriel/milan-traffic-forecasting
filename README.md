# Mobile Network Traffic Forecasting Milan Telecom Dataset

Comparative study of three sequential models **SARIMAX with Fourier terms**, **LSTM** and
**TCN** for one-step-ahead forecasting of 10-minute Internet traffic in the three busiest
areas of Milan (squares 5161, 5059, 5259), evaluated on the week of 16–22 December 2013.

**Research question:** How do different sequential models compare for one-step-ahead mobile
network traffic forecasting, and how does their performance vary across geographical areas
with different traffic characteristics?

- Demo Video: *https://youtu.be/3TILXM5ivdE*

## Repository structure


milan-traffic-forecasting/
├── configs/
│   └── final_params.json      # hyperparameters selected during tuning
├── scripts/
│   ├── 01_prepare_data.py     # memory-efficient loading + memory benchmark
│   ├── 02_eda.py              # exploratory analysis figures and statistics
│   ├── 03_tune.py             # one tuning experiment -> results/experiments.csv
│   ├── 04_final.py            # final test-week evaluation (3 areas, 3 seeds)
│   └── 05_diagnostics.py      # error analysis + post-hoc weekly-lag ablation
├── src/
│   ├── config.py              # paths, dates, splits, constants
│   ├── data.py                # raw file readers, matrix builder
│   ├── processed.py           # loading the processed matrix (memory-mapped)
│   ├── features.py            # splits, calendar features, scaling, windows, metrics
│   ├── runner.py              # common train/forecast interface for all models
│   └── models/
│       ├── sarimax_fourier.py # SARIMAX with Fourier seasonal terms
│       └── neural.py          # LSTM, TCN, training loop with early stopping
├── results/                   # CSV/JSON outputs (experiment log, metrics, timing)
├── figures/                   # all figures used in the report
├── data/                      # NOT in git (see "Data")
└── requirements.txt

## Setup

Tested on Windows 11, Python 3.11.9, CPU only (Intel Core i5-1135G7, 4 cores / 8 threads, 8 GB RAM).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

Exact package versions used for the reported results are listed in `requirements-lock.txt`
(`pip install -r requirements-lock.txt` for an exact reproduction).

## Data

The data is not stored in this repository (about 20 GB raw).

1. Download the 62 daily files `sms-call-internet-mi-2013-11-01.txt` … `2014-01-01.txt` of
   *Telecommunications – SMS, Call, Internet – MI* from Harvard Dataverse
   (doi: [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)) into `data/raw/`.
2. Optional: download `milano-grid.geojson` (doi: [10.7910/DVN/QJWLFU](https://doi.org/10.7910/DVN/QJWLFU))
   to `data/milano-grid.geojson` to obtain map coordinates in the EDA.

## Reproducing the results

Run all commands from the repository root with the virtual environment active.
Times are approximate for the hardware above.

| Step | Command | Time | Main outputs |
|---|---|---|---|
| 1. Preprocessing | `python -m scripts.01_prepare_data` | ~7 min | `data/processed/`, `results/memory_benchmark.csv`, `results/data_summary.json` |
| 2. Exploratory analysis | `python -m scripts.02_eda` | ~2 min | `figures/fig1–fig5`, `results/eda_summary.json` |
| 3. Tuning (optional) | see below | ~2 h total | `results/experiments.csv`, `figures/tuning/` |
| 4. Final evaluation | `python -m scripts.04_final` | ~25 min | `results/final/`, `figures/final/` |
| 5. Diagnostics | `python -m scripts.05_diagnostics --ablation lstm tcn` | ~20 min | `results/final/diagnostics.json`, `results/final/ablation_weekly_lag.csv` |

Steps 4 and 5 use the hyperparameters in `configs/final_params.json`, so the tuning step does
not need to be repeated to reproduce the final results.

### Tuning

Each call trains on the training split, evaluates on the validation week for square 5161 and
appends one row to `results/experiments.csv` (the full log of the 34 experiments is included).

```bash
python -m scripts.03_tune --model lstm --set window=36 --set lr=0.0003 --seed 1 --note "reason"
python -m scripts.03_tune --model sarimax --grid p=1,2,3 --grid q=0,1,2 --note "reason"
```

`--model` is one of `persistence`, `seasonal_naive`, `sarimax`, `lstm`, `tcn`; `--set` fixes a
parameter, `--grid` tries every combination, `--seed` sets the random seed.

## Experimental setup

| | |
|---|---|
| Target | Internet activity per 10-minute interval, one step ahead |
| Train / validation / test | 1 Nov – 8 Dec / 9 – 15 Dec / 16 – 22 Dec 2013 (Europe/Rome time) |
| Areas | 5161, 5059, 5259 (highest total traffic) |
| Reference models | persistence (last value), seasonal naive (same time last week) |
| Metrics | MAE, RMSE, MAPE on the original scale |
| Neural networks | 3 seeds (42, 1, 2) per area; mean ± std reported, median-seed run plotted |
| Random seed | 42 (all deterministic parts) |

## Main results (test week, MAE; neural networks mean over 3 seeds)

| Model | 5161 | 5059 | 5259 |
|---|---|---|---|
| Persistence | 92.8 | 81.5 | 76.0 |
| Seasonal naive | 300.8 | 260.0 | 210.2 |
| **SARIMAX-Fourier** | **77.9** | **65.6** | **67.0** |
| LSTM | 109.0 ± 8.5 | 100.0 ± 11.1 | 74.0 ± 1.1 |
| TCN | 112.2 ± 9.8 | 97.5 ± 11.9 | 81.4 ± 1.9 |

Full tables (MAE, RMSE, MAPE), timing statistics and the hardware description are in `results/final/`.

## Author

Gabriel Sezibera Tuyisingize Formative 1 Assignment , September 2026.

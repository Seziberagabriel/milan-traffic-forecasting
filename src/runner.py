"""One entry point that trains any model on the training split and returns a function
producing one-step-ahead forecasts for an evaluation range (validation or test)."""
import json
import platform
import subprocess

import numpy as np
import psutil

from . import config as C
from .features import build_nn_inputs

MAIN_MODELS = ["sarimax", "lstm", "tcn"]
REFERENCE_MODELS = ["persistence", "seasonal_naive"]

DEFAULTS = {
    "persistence": {"lag": 1},
    "seasonal_naive": {"lag": C.WEEK},
    "sarimax": {"p": 2, "q": 1, "daily_k": 6, "weekly_k": 3, "holiday": True,
                "log": True, "maxiter": 200},
    "lstm": {"window": 144, "hidden": 64, "layers": 1, "dropout": 0.0, "lr": 1e-3,
             "batch": 128, "max_epochs": 40, "patience": 6, "weekly_lag": True},
    "tcn": {"window": 144, "channels": 32, "kernel": 3, "levels": 6, "dropout": 0.1,
            "lr": 1e-3, "batch": 128, "max_epochs": 40, "patience": 6, "weekly_lag": True},
}


def run(model_name, params, y, index, splits, eval_start, eval_end, seed=C.SEED):
    """Returns (predict_fn, info). predict_fn() -> forecasts in original units for
    positions [eval_start, eval_end). Calling it repeatedly lets us time inference."""
    p = {**DEFAULTS[model_name], **(params or {})}
    y = np.asarray(y, dtype=np.float64)

    if model_name in REFERENCE_MODELS:
        lag = p["lag"]
        return (lambda: y[eval_start - lag:eval_end - lag].copy()), {"train_seconds": 0.0, "params": p}

    if model_name == "sarimax":
        from .models.sarimax_fourier import fit_predict
        predict_fn, info = fit_predict(y, index, splits, eval_start, eval_end, p)
        info["params"] = p
        return predict_fn, info

    # neural models
    import torch
    from .models import neural
    if p["window"] > C.MAX_WINDOW:
        raise ValueError(f"window must be <= {C.MAX_WINDOW}")
    X, z, scaler = build_nn_inputs(y, index, splits, p["weekly_lag"])
    train_t = np.arange(C.FIRST_NN_TARGET, splits["train_end"])
    val_t = np.arange(splits["val_start"], splits["val_end"])
    eval_t = np.arange(eval_start, eval_end)

    neural.set_seed(seed)
    model = neural.build_model(model_name, X.shape[1], p)
    info = neural.train(model, X, z, train_t, val_t, p, seed)
    info["n_params"] = int(sum(q.numel() for q in model.parameters()))
    info["n_train_samples"] = int(len(train_t))
    if model_name == "tcn":
        info["receptive_field"] = model.receptive_field
    info["params"] = p
    neural.predict(model, X, eval_t[:8], p["window"])      # warm-up before any timing
    return (lambda: scaler.inverse(neural.predict(model, X, eval_t, p["window"]))), info


def hardware_info():
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            cpu = out or cpu
        except Exception:
            pass
    info = {"cpu_model": cpu, "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1),
            "os": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        info.update(torch=torch.__version__, torch_threads=torch.get_num_threads(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)")
    except ImportError:
        pass
    return info


def to_json(obj):
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))

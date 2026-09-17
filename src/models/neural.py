"""Models 2 and 3: LSTM (recurrent) and TCN (dilated causal convolutions), CPU-friendly."""
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from ..features import gather_windows


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class LSTMForecaster(nn.Module):
    """Stacked LSTM; the last hidden state is mapped to the next-step value."""

    def __init__(self, n_features, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                      # x: (batch, window, features)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


class CausalBlock(nn.Module):
    """Residual block of two dilated causal convolutions (Bai et al., 2018)."""

    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        self.chop = (kernel - 1) * dilation     # remove right padding -> no future leakage
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=self.chop, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=self.chop, dilation=dilation)
        self.act, self.drop = nn.ReLU(), nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(self.act(self.conv1(x)[..., :-self.chop]))
        h = self.drop(self.act(self.conv2(h)[..., :-self.chop]))
        return self.act(h + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(self, n_features, channels=32, kernel=3, levels=6, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(*[
            CausalBlock(n_features if i == 0 else channels, channels, kernel, 2 ** i, dropout)
            for i in range(levels)])
        self.head = nn.Linear(channels, 1)
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** levels - 1)

    def forward(self, x):                      # x: (batch, window, features)
        h = self.net(x.transpose(1, 2))        # -> (batch, channels, window)
        return self.head(h[:, :, -1]).squeeze(-1)


def build_model(name, n_features, p):
    if name == "lstm":
        return LSTMForecaster(n_features, p["hidden"], p["layers"], p["dropout"])
    if name == "tcn":
        return TCNForecaster(n_features, p["channels"], p["kernel"], p["levels"], p["dropout"])
    raise ValueError(name)


@torch.no_grad()
def predict(model, X, targets, window, batch=512):
    model.eval()
    out = [model(torch.from_numpy(gather_windows(X, targets[s:s + batch], window))).numpy()
           for s in range(0, len(targets), batch)]
    return np.concatenate(out)


def train(model, X, z, train_t, val_t, p, seed):
    """Adam + MSE on the scaled log target, gradient clipping, early stopping on
    validation MSE (best weights restored)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    loss_fn = nn.MSELoss()
    best, best_state, best_epoch, wait, history = np.inf, None, 0, 0, []

    t0 = time.perf_counter()
    for epoch in range(1, p["max_epochs"] + 1):
        model.train()
        perm = rng.permutation(train_t)
        total = 0.0
        for s in range(0, len(perm), p["batch"]):
            b = perm[s:s + p["batch"]]
            xb = torch.from_numpy(gather_windows(X, b, p["window"]))
            yb = torch.from_numpy(z[b])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        val_mse = float(np.mean((predict(model, X, val_t, p["window"]) - z[val_t]) ** 2))
        history.append({"epoch": epoch, "train_mse": total / len(perm), "val_mse": val_mse})
        if val_mse < best - 1e-6:
            best, best_state, best_epoch, wait = val_mse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= p["patience"]:
                break
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return {"train_seconds": train_seconds, "epochs_run": epoch, "best_epoch": best_epoch,
            "seconds_per_epoch": train_seconds / epoch, "best_val_mse_scaled": best,
            "history": history}

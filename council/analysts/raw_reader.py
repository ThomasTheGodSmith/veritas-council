"""Analyst 7, Raw Reader (PLAN.md §1b-7): our own small 1D CNN that "reads the chart".

Input for a row (ticker, day t): the last 120 trading days up to and including t, two channels:
  * price:  adj_close / adj_close on the window's first day  (so every window starts at 1.0)
  * volume: ln(1 + volume), standardised within the window   ((x - mean) / std)
No indicators. The network has to find any useful shapes itself.

Network (trained FROM SCRATCH as a classifier, p = sigmoid(CNN(window)), cross-entropy loss):
  conv(2->16, width 5) -> ReLU -> max-pool /2      (120 -> 60 days)
  conv(16->32, width 5) -> ReLU -> max-pool /2     (60 -> 30)
  conv(32->32, width 5) -> ReLU -> average over time (global pooling)
  dense(32->16) -> ReLU -> dense(16->1)
The target is the panel's y (did the price rise from the next open to 20 days later).

Difference from the Pattern Oracle: that one *writes* possible futures; this one looks at the
chart and answers up/down directly.

No look-ahead: a row's window only contains prices dated <= the row's date.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from council.analysts.base import Analyst, Context

SEED = 0


def _window_days() -> int:
    cfg = yaml.safe_load(Path("config/agents.yaml").read_text(encoding="utf-8"))
    return int(cfg.get("raw_reader", {}).get("window_days", 120))


class ChartCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 32, 5, padding=2)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 2 channels, 120 days) -> (batch,) score; sigmoid(score) = P(up)."""
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x)).mean(dim=2)      # global average over time
        return self.fc2(self.drop(F.relu(self.fc1(x)))).squeeze(1)


def scale_windows(price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """(n, days) raw windows -> (n, 2, days) network input, each window scaled on its own."""
    p = price / price[:, :1]
    lv = np.log1p(volume)
    lv = (lv - lv.mean(1, keepdims=True)) / (lv.std(1, keepdims=True) + 1e-6)
    return np.stack([p, lv], axis=1).astype(np.float32)


class RawReader(Analyst):
    name = "raw_reader"
    features: list[str] = []

    def __init__(self, epochs: int = 3, window: int | None = None):
        self.window = window or _window_days()
        self.epochs = epochs
        self.batch_size = 256
        self.state: dict | None = None     # trained weights (a state_dict; pickles cleanly)
        self._net: ChartCNN | None = None

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_net"] = None
        return s

    @property
    def net(self) -> ChartCNN:
        if self._net is None:
            self._net = ChartCNN()
            self._net.load_state_dict(self.state)
            self._net.eval()
        return self._net

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # -------------------------------------------------------------------------- windows
    def windows(self, rows: pd.DataFrame, ctx: Context) -> tuple[np.ndarray, np.ndarray]:
        """Network input for every row (zeros where there are under 120 days of history) and a
        mask saying which rows had enough history."""
        X = np.zeros((len(rows), 2, self.window), dtype=np.float32)
        ok = np.zeros(len(rows), dtype=bool)
        prices = ctx.prices[ctx.prices.ticker.isin(set(rows.ticker))]
        by_ticker = {t: g.sort_values("date") for t, g in prices.groupby("ticker")}
        row_dates = pd.to_datetime(rows["date"]).to_numpy("datetime64[ns]")
        for t, idx in rows.groupby("ticker").indices.items():
            if t not in by_ticker:
                continue
            g = by_ticker[t]
            dates = g["date"].to_numpy("datetime64[ns]")
            c = g["adj_close"].ffill().to_numpy(float)
            v = g["volume"].fillna(0).to_numpy(float)
            end = np.searchsorted(dates, row_dates[idx], side="right")   # days dated <= t
            good = end >= self.window
            if not good.any():
                continue
            starts = end[good] - self.window
            pw = np.lib.stride_tricks.sliding_window_view(c, self.window)[starts]
            vw = np.lib.stride_tricks.sliding_window_view(v, self.window)[starts]
            X[idx[good]] = scale_windows(pw, vw)
            ok[idx[good]] = True
        ok &= np.isfinite(X).all(axis=(1, 2))
        return X, ok

    # ------------------------------------------------------------------------------ fit
    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        torch.set_num_threads(4)
        df = panel.dropna(subset=["y"]).reset_index(drop=True)
        self._learn_payoffs(df)
        self.base_rate = float(df["y"].mean())       # used for rows with too little history
        X, ok = self.windows(df, ctx)
        X, y = torch.from_numpy(X[ok]), torch.from_numpy(df["y"].to_numpy(np.float32)[ok])
        net = ChartCNN()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        print(f"  raw_reader: {len(X):,} windows, {sum(p.numel() for p in net.parameters()):,} parameters")
        t0 = time.time()
        for ep in range(self.epochs):
            net.train()
            order = torch.from_numpy(rng.permutation(len(X)))
            total = 0.0
            for i in range(0, len(X), self.batch_size):
                b = order[i:i + self.batch_size]
                loss = F.binary_cross_entropy_with_logits(net(X[b]), y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            # a model that always says the base rate scores this:
            br = self.base_rate
            base = -(br * np.log(br) + (1 - br) * np.log(1 - br))
            print(f"  raw_reader epoch {ep + 1}/{self.epochs}: loss {total / len(X):.4f} "
                  f"(base-rate guess = {base:.4f}), {time.time() - t0:.0f}s")
        net.eval()
        self.state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        self._net = net

    @torch.no_grad()
    def _predict_windows(self, X: np.ndarray) -> np.ndarray:
        torch.set_num_threads(4)
        out = [torch.sigmoid(self.net(torch.from_numpy(X[i:i + 4096]))).numpy() for i in range(0, len(X), 4096)]
        return np.concatenate(out) if out else np.zeros(0)

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        rows = panel.reset_index(drop=True)
        X, ok = self.windows(rows, ctx)
        p = np.full(len(rows), self.base_rate)
        if ok.any():
            p[ok] = self._predict_windows(X[ok])
        return p

    # ------------------------------------------------------------------------- explanations
    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        """Occlusion: blank out part of the chart (flat price, average volume) and see how much
        P(up) moves. A big change means that part of the chart mattered."""
        X, ok = self.windows(row.to_frame().T.reset_index(drop=True), ctx)
        if not ok[0]:
            return [("not enough history: base rate used", self.base_rate)]
        last20, earlier = X.copy(), X.copy()
        last20[0, 0, -20:] = last20[0, 0, -21]       # freeze the price for the last 20 days
        last20[0, 1, -20:] = 0.0
        earlier[0, 0, :-20] = earlier[0, 0, -20]     # freeze the price before that
        earlier[0, 1, :-20] = 0.0
        p, p_last, p_early = self._predict_windows(np.concatenate([X, last20, earlier]))
        out = [("last 20 days of the chart", float(p - p_last)),
               ("earlier 100 days of the chart", float(p - p_early))]
        return sorted(out, key=lambda kv: -abs(kv[1]))

    def key_numbers(self, row: pd.Series, ctx: Context) -> dict[str, float]:
        X, ok = self.windows(row.to_frame().T.reset_index(drop=True), ctx)
        if not ok[0]:
            return {}
        return {"chart_change_120d_pct": float((X[0, 0, -1] - 1) * 100),
                "probability_up": float(self._predict_windows(X)[0])}


def build() -> RawReader:
    return RawReader()

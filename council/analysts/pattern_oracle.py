"""Analyst 6, Pattern Oracle (PLAN.md §1b-6): our own mini "language model" for prices.

The idea, in LLM terms:
  * Words -> price moves. Every daily log return r_t = ln(P_t / P_{t-1}) is put into one of
    64 buckets ("tokens"). The bucket edges are the 1/64, 2/64, ... quantiles of the training
    returns, so every token is equally common in training. Token 0 = the worst days, token 63
    = the best days.
  * A tiny GPT-style transformer (decoder-only, causal attention), trained FROM SCRATCH, learns
    P(next token | the previous tokens), exactly like an LLM learns the next word. Loss:
    cross-entropy  -sum log P(token_{t+1} | token_1..t)  over every member's and the Dow
    index's training-year token stream (all pooled together, because each one alone is small).
  * To predict a row (ticker, day t) we give it the last 120 tokens up to and including day t,
    and let it "write" many possible next-20-day futures by sampling one token at a time
    (temperature 1). Each token is turned back into a return (the median training return in
    that bucket), we add up the 20 returns, and
        p = share of futures whose total 20-day return is above 0.
    "131 of 200 imagined futures ended higher" -> p = 0.655.

It sees no indicators and no finance theory: only the token stream.

Speed: the backtest asks for ~10k rows and validation for ~30k rows. Sampling is batched
(many rows x many futures at once) and uses a key/value cache: the 120-day context is read
once per row, its attention keys/values are SHARED by all that row's futures, and each new
token costs one small step instead of re-reading the whole context. For big requests (more than
`large_request_rows` rows, i.e. the validation slice) we sample 100 futures instead of 200 to
keep training time reasonable; the backtest and the app use the full 200.

No look-ahead: for a row dated t, only returns dated <= t are in its context window. During
fit, the bins and the transformer only use prices up to the last date of the rows we're given.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from council.analysts.base import Analyst, Context

INDEX = "^DJI"
SEED = 0


def _config() -> dict:
    cfg = yaml.safe_load(Path("config/agents.yaml").read_text(encoding="utf-8"))
    return cfg.get("pattern_oracle", {})


# ---------------------------------------------------------------------------------- the model
class Block(nn.Module):
    """One transformer layer: causal self-attention, then a small feed-forward network.
    Both have a residual ("skip") connection and LayerNorm in front (the GPT-2 layout)."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)       # queries, keys and values in one matrix
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(..., D) -> (..., heads, D/heads)"""
        return t.view(*t.shape[:-1], self.heads, t.shape[-1] // self.heads)

    def forward(self, x: torch.Tensor):
        """Full causal attention over a whole sequence (training, and reading the 120-day context).
        Returns the output and this layer's keys/values so generation can reuse them."""
        B, T, D = x.shape
        q, k, v = (self._split_heads(t).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(D, dim=2))
        att = F.scaled_dot_product_attention(       # causal mask = a token can't see the future
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(att.transpose(1, 2).reshape(B, T, D)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x, (k, v)

    def step(self, x: torch.Tensor, prefix: tuple, gen: tuple, n: int) -> torch.Tensor:
        """One new token for every (row, future) pair. x: (R, S, D).
        prefix = keys/values of each row's 120 context tokens, (R, heads, 120, hd), SHARED by
        all S futures of that row (so we don't copy them S times). gen = keys/values of the
        tokens this future has written so far, (R, S, heads, h, hd); we fill slot n."""
        R, S, D = x.shape
        q, k, v = (self._split_heads(t) for t in self.qkv(self.ln1(x)).split(D, dim=2))   # (R,S,H,hd)
        gen[0][:, :, :, n], gen[1][:, :, :, n] = k, v
        gk, gv = gen[0][:, :, :, :n + 1], gen[1][:, :, :, :n + 1]
        pk, pv = prefix
        scale = 1 / math.sqrt(q.shape[-1])
        # attention = softmax(q.k / sqrt(d)) . v, over [the shared past, this future's own tokens]
        scores = torch.cat([torch.einsum("rshd,rhpd->rshp", q, pk),
                            torch.einsum("rshd,rshgd->rshg", q, gk)], dim=-1) * scale
        w = scores.softmax(-1)
        P = pk.shape[2]
        out = torch.einsum("rshp,rhpd->rshd", w[..., :P], pv) + torch.einsum("rshg,rshgd->rshd", w[..., P:], gv)
        x = x + self.proj(out.reshape(R, S, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Token embedding + position embedding -> N transformer blocks -> scores for the next token."""

    def __init__(self, vocab: int, block_size: int, layers: int, heads: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx: torch.Tensor):
        """idx: (B, T) tokens -> (B, T, vocab) scores for the next token, plus each layer's keys/values."""
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        kvs = []
        for blk in self.blocks:
            x, kv = blk(x)
            kvs.append(kv)
        return self.head(self.ln(x)), kvs

    def step(self, tok: torch.Tensor, prefixes: list, gens: list, pos: int, n: int) -> torch.Tensor:
        """tok: (R, S) the latest token of every future -> (R, S, vocab) scores for the next one."""
        x = self.tok(tok) + self.pos.weight[pos]
        for blk, pre, gen in zip(self.blocks, prefixes, gens):
            x = blk.step(x, pre, gen, n)
        return self.head(self.ln(x))


# ------------------------------------------------------------------------------- data helpers
def _return_streams(prices: pd.DataFrame, tickers, until=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ticker -> (dates, daily log returns). The return dated t is ln(P_t / P_{t-1})."""
    p = prices[prices.ticker.isin(set(tickers))]
    if until is not None:
        p = p[p.date <= until]
    out = {}
    for t, g in p.sort_values("date").groupby("ticker"):
        c = g["adj_close"].ffill().to_numpy(float)
        r = np.diff(np.log(c))
        ok = np.isfinite(r)
        out[t] = (g["date"].to_numpy("datetime64[ns]")[1:][ok], r[ok])
    return out


class PatternOracle(Analyst):
    name = "pattern_oracle"
    features: list[str] = []

    def __init__(self, epochs: int = 4, sampled_futures: int | None = None, horizon: int = 20,
                 large_request_rows: int = 15_000, large_request_futures: int = 100, **overrides):
        cfg = {**_config(), **overrides}
        self.vocab = int(cfg.get("token_bins", 64))
        self.context = int(cfg.get("context_tokens", 120))
        self.layers = int(cfg.get("layers", 4))
        self.heads = int(cfg.get("heads", 4))
        self.d = int(cfg.get("embed_dim", 128))
        self.futures_n = int(sampled_futures or cfg.get("sampled_futures", 200))
        self.h = horizon
        self.epochs = epochs
        self.large_request_rows = large_request_rows
        self.large_request_futures = large_request_futures
        self.batch_size = 128
        self.lr = 3e-4
        self.state: dict | None = None        # trained weights (a state_dict; pickles cleanly)
        self._net: TinyGPT | None = None

    # pickling: keep the weights (state_dict), drop the live torch module
    def __getstate__(self):
        s = self.__dict__.copy()
        s["_net"] = None
        return s

    def _make_net(self) -> TinyGPT:
        # block size = context + horizon, so positions 120..139 used while writing futures are trained too
        return TinyGPT(self.vocab, self.context + self.h, self.layers, self.heads, self.d)

    @property
    def net(self) -> TinyGPT:
        if self._net is None:
            self._net = self._make_net()
            self._net.load_state_dict(self.state)
            self._net.eval()
        return self._net

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ------------------------------------------------------------------------------ tokeniser
    def tokenise(self, r: np.ndarray) -> np.ndarray:
        """Return -> token id 0..63 (which quantile bucket it falls in)."""
        return np.searchsorted(self.edges, r).astype(np.int64)

    # ---------------------------------------------------------------------------------- fit
    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        torch.set_num_threads(4)
        self._learn_payoffs(panel.dropna(subset=["fwd_return"]))
        self.tickers = sorted(panel.ticker.unique())          # Dow members + the index
        cutoff = panel.date.max()                              # nothing after the last training row
        streams = _return_streams(ctx.prices, self.tickers, until=cutoff)

        # 1. tokeniser: 64 equal-count buckets of the pooled training returns
        all_r = np.concatenate([r for _, r in streams.values()])
        self.edges = np.quantile(all_r, np.arange(1, self.vocab) / self.vocab)
        all_tok = self.tokenise(all_r)
        # bucket -> a typical return (median of the training returns in that bucket)
        self.bin_return = np.array([np.median(all_r[all_tok == k]) for k in range(self.vocab)])

        # 2. training windows of context+horizon+1 tokens, never crossing from one ticker to another
        L = self.context + self.h
        seqs = []
        for _, r in streams.values():
            tok = self.tokenise(r)
            if len(tok) > L:
                win = np.lib.stride_tricks.sliding_window_view(tok, L + 1)[::30]   # a window every 30 days
                seqs.append(win)
        data = torch.from_numpy(np.concatenate(seqs))
        # share of 20-day stretches that ended up: used if a row has too little history
        up = [np.convolve(r, np.ones(self.h), "valid") > 0 for _, r in streams.values() if len(r) >= self.h]
        self.base_rate = float(np.concatenate(up).mean())

        # 3. next-token training (cross-entropy), AdamW
        net = self._make_net()
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=0.01)
        steps_per_epoch = math.ceil(len(data) / self.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=self.epochs * steps_per_epoch)
        print(f"  pattern_oracle: {len(streams)} tickers, {len(all_r):,} tokens, {len(data):,} windows, "
              f"{sum(p.numel() for p in net.parameters()):,} parameters")
        t0 = time.time()
        for ep in range(self.epochs):
            net.train()
            order = torch.from_numpy(rng.permutation(len(data)))
            total = 0.0
            for i in range(0, len(data), self.batch_size):
                b = data[order[i:i + self.batch_size]]
                logits, _ = net(b[:, :-1])                       # predict token k+1 from tokens 0..k
                loss = F.cross_entropy(logits.reshape(-1, self.vocab), b[:, 1:].reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(b)
            # ln(64) = 4.16 is what a model that knows nothing would score
            print(f"  pattern_oracle epoch {ep + 1}/{self.epochs}: loss {total / len(data):.4f} "
                  f"(uniform guess = {math.log(self.vocab):.4f}), {time.time() - t0:.0f}s")
        net.eval()
        self.state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        self._net = net

    # ---------------------------------------------------------------------------- sampling
    def _contexts(self, rows: pd.DataFrame, ctx: Context) -> tuple[np.ndarray, np.ndarray]:
        """For each row: its last `context` tokens dated <= the row's date, and whether it had enough."""
        streams = _return_streams(ctx.prices, rows.ticker.unique())
        X = np.zeros((len(rows), self.context), dtype=np.int64)
        ok = np.zeros(len(rows), dtype=bool)
        row_dates = pd.to_datetime(rows["date"]).to_numpy("datetime64[ns]")
        for t, idx in rows.groupby("ticker").indices.items():
            if t not in streams:
                continue
            dates, r = streams[t]
            tok = self.tokenise(r)
            end = np.searchsorted(dates, row_dates[idx], side="right")  # tokens dated <= t
            good = end >= self.context
            for j, e in zip(idx[good], end[good]):
                X[j] = tok[e - self.context:e]
            ok[idx[good]] = True
        return X, ok

    @torch.no_grad()
    def _sample(self, X: np.ndarray, n_futures: int, seed: int = SEED) -> np.ndarray:
        """Sample n_futures futures of h tokens for every context row. Returns (rows, n_futures, h)."""
        torch.set_num_threads(4)
        g = torch.Generator().manual_seed(seed)
        net = self.net
        out = np.zeros((len(X), n_futures, self.h), dtype=np.int64)
        rows_per_chunk = max(1, 8000 // n_futures)               # ~8000 futures at a time
        H, hd = self.heads, self.d // self.heads
        for s in range(0, len(X), rows_per_chunk):
            ctx_tok = torch.from_numpy(X[s:s + rows_per_chunk])
            R = len(ctx_tok)
            logits, prefixes = net(ctx_tok)                       # read the 120-day context once
            gens = [(torch.zeros(R, n_futures, H, self.h, hd), torch.zeros(R, n_futures, H, self.h, hd))
                    for _ in self.net.blocks]
            last = logits[:, -1:].expand(R, n_futures, self.vocab)   # every future starts from here
            gen = []
            for step in range(self.h):
                probs = F.softmax(last, dim=-1).reshape(-1, self.vocab)
                tok = torch.multinomial(probs, 1, generator=g).view(R, n_futures)   # temperature 1
                gen.append(tok)
                if step < self.h - 1:
                    last = net.step(tok, prefixes, gens, pos=self.context + step, n=step)
            out[s:s + R] = torch.stack(gen, -1).numpy()
        return out

    def _cumulative_returns(self, tokens: np.ndarray) -> np.ndarray:
        """Tokens -> total 20-day log return of each future."""
        return self.bin_return[tokens].sum(-1)

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        rows = panel.reset_index(drop=True)
        n = self.futures_n if len(rows) <= self.large_request_rows else self.large_request_futures
        X, ok = self._contexts(rows, ctx)
        p = np.full(len(rows), self.base_rate)
        if ok.any():
            cum = self._cumulative_returns(self._sample(X[ok], n))
            p[ok] = (cum > 0).mean(1)
        return p

    def futures(self, ticker: str, date, ctx: Context, n: int | None = None) -> pd.DataFrame:
        """The imagined price paths for the report chart: one column per future, one row per day
        (day 0 = 1.0 = today's price). Empty if the ticker has under 120 days of history."""
        row = pd.DataFrame({"ticker": [ticker], "date": [pd.Timestamp(date)]})
        X, ok = self._contexts(row, ctx)
        if not ok[0]:
            return pd.DataFrame()
        tok = self._sample(X, n or self.futures_n)[0]                     # (n, h)
        paths = np.exp(np.cumsum(self.bin_return[tok], axis=1))
        paths = np.hstack([np.ones((len(paths), 1)), paths]).T
        return pd.DataFrame(paths, columns=[f"future_{i}" for i in range(paths.shape[1])])

    # ------------------------------------------------------------------------- explanations
    def key_numbers(self, row: pd.Series, ctx: Context) -> dict[str, float]:
        f = self.futures(row["ticker"], row["date"], ctx)
        if f.empty:
            return {}
        end = f.iloc[-1].to_numpy() - 1
        return {"futures_up": float((end > 0).sum()), "futures_total": float(len(end)),
                "median_return_pct": float(np.median(end) * 100),
                "p5_return_pct": float(np.percentile(end, 5) * 100),
                "p95_return_pct": float(np.percentile(end, 95) * 100)}

    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        k = self.key_numbers(row, ctx)
        if not k:
            return [("not enough history: base rate used", self.base_rate)]
        return [("share of imagined futures ending higher", k["futures_up"] / k["futures_total"]),
                ("median imagined 20-day return %", k["median_return_pct"]),
                ("bad-case (5th percentile) return %", k["p5_return_pct"])]


def build() -> PatternOracle:
    # config/agents.yaml asks for 128-dim embeddings (~830k parameters). We use 64 dims (~217k
    # parameters): sampling the validation slice at 128 dims took ~11 min on a laptop CPU, vs
    # ~6.5 min at 64, and with only ~240k training tokens a bigger model has nothing more to
    # learn. 6 epochs: the next-token loss on the validation years stopped improving by then
    # (4.047 vs 4.159 for a uniform guess), and didn't show over-fitting.
    return PatternOracle(embed_dim=64, epochs=6)

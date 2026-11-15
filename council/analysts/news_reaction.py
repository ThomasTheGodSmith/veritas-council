"""Analyst 5, News-Reaction: our own text model, trained from scratch (PLAN.md §1b-5).

No pre-trained model of any kind: no FinBERT, no VADER, no word embeddings. The model learns its
own "good news" and "bad news" words from what prices actually did after the headlines.

TIMING (no look-ahead). A panel row dated t is a decision taken after day t's close and acted on
at the next open. To be safe we only use headlines dated t-1 or earlier (`LAG_DAYS = 1`):
Reddit/Kaggle days are UTC calendar days and some "day t" headlines are posted after the US
close, so we simply skip day t. That satisfies the Historian rule "date < decision_date" with a
day to spare. The synthetic test tests/test_news_reaction.py checks this.

WHAT A ROW "READS"
  company block  that company's headlines dated t-3 .. t-1 (3 calendar days covers weekends).
                 For the Dow index row: every member's headlines in that window, weighted by the
                 member's Dow price weight on day t (PLAN.md §0b).
  market block   the top-25 world headlines of the most recent news day in t-3 .. t-1.
Each block is a TF-IDF vector (vocabulary fitted on training headlines only, 1-2 word phrases,
sublinear term counts), scaled to length 1. The two blocks sit side by side, so the model learns
separate weights for "company words" and "market words".

THE THREE PARTS, combined once (news is never counted twice):
  1. p_text    = sigmoid(w . tfidf)  logistic regression on the row's 20-day label y.
  2. retrieved = similarity-weighted average 5-day abnormal return of the 20 most similar past
                 days for the same ticker (cosine similarity), using only days whose outcome was
                 already known (label_known <= t). Abnormal return AR = r - (alpha + beta * r_DJI),
                 alpha and beta from a 250-day regression on PAST days only. For the index itself
                 AR would be ~0, so its "CAR" is just its own 5-day return.
  3. news_volume_z = how unusually many company headlines there are (log count, z-scored per ticker).
  p = sigmoid(a + b*logit(p_text) + c*retrieved + d*news_volume_z), with a..d fitted on
  OUT-OF-FOLD p_text (the text model never grades its own homework, see `fit`).

NO NEWS: if a row has no headlines at all, we return the training base rate for its group
(index or member) and set has_news = 0 in key_numbers, so the Mediator can learn to ignore it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from council.analysts.base import Analyst, Context
from council.features import INDEX, member_mask

LAG_DAYS = 1        # newest headlines allowed: dated t - 1 day
WINDOW_DAYS = 3     # oldest headlines allowed: dated t - 3 days
TOP_K = 20          # similar past days used for `retrieved`
CAR_DAYS = 5        # abnormal return measured over the next 5 trading days
BETA_DAYS = 250     # rolling regression length for alpha and beta
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]   # text-model regularisation, picked inside the fit years


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class NewsReaction(Analyst):
    name = "news_reaction"
    features = ["has_news", "n_headlines", "n_company_headlines", "n_market_headlines",
                "p_text", "retrieved_return", "news_volume_z"]

    # ================================================================== building the text vectors
    def _news_in_window(self, panel: pd.DataFrame, ctx: Context):
        """Only the news that can matter for these rows (dated between first_t-3 and last_t-1)."""
        lo = panel.date.min() - pd.Timedelta(days=WINDOW_DAYS)
        hi = panel.date.max() - pd.Timedelta(days=LAG_DAYS)
        empty = pd.DataFrame(columns=["date", "ticker", "headline"])
        mk = ctx.market_news if ctx.market_news is not None else empty
        co = ctx.company_news if ctx.company_news is not None else empty
        mk = mk[(mk.date >= lo) & (mk.date <= hi)]
        co = co[(co.date >= lo) & (co.date <= hi) & co.ticker.isin(self._known_tickers(panel, ctx))]
        return mk, co

    @staticmethod
    def _known_tickers(panel: pd.DataFrame, ctx: Context) -> set[str]:
        tickers = set(panel.ticker) - {INDEX}
        if (panel.ticker == INDEX).any():
            tickers.add(INDEX)            # headlines tagged with the index itself (single-stock runs)
        if (panel.ticker == INDEX).any() and ctx.membership is not None:
            tickers |= set(ctx.membership.ticker.dropna())   # the index reads its members' news
        return tickers

    def _index_weights(self, dates: pd.Series, ctx: Context) -> pd.DataFrame:
        """Dow price weight of every member on each date: raw price / sum of members' raw prices."""
        px = ctx.prices[ctx.prices.date.isin(set(dates)) & ctx.prices.ticker.ne(INDEX)][["date", "ticker", "raw_close"]]
        if ctx.membership is None or px.empty:   # typed, so later merges on date still work
            return pd.DataFrame({"date": pd.Series([], dtype=pd.Series(dates).dtype),
                                 "ticker": pd.Series([], dtype=object), "w": pd.Series([], dtype=float)})
        px = px[member_mask(px, ctx.membership).values]
        px = px.assign(w=px.raw_close / px.groupby("date").raw_close.transform("sum"))
        return px[["date", "ticker", "w"]]

    def _vectors(self, panel: pd.DataFrame, ctx: Context) -> dict:
        """For every row: company TF-IDF block, market block, headline counts and sample headlines."""
        rows = panel[["date", "ticker"]].reset_index(drop=True).assign(i=lambda d: np.arange(len(d)))
        n = len(rows)
        mk, co = self._news_in_window(panel, ctx)
        F = len(self.vectorizer.vocabulary_)

        # ---- company block: one document per (ticker, day), joined into each row's window
        cdocs = co.groupby(["ticker", "date"]).headline.agg(list).reset_index()
        cdocs["j"] = np.arange(len(cdocs))
        cdocs["count"] = cdocs.headline.map(len)
        Vc = (self.vectorizer.transform(cdocs.headline.map(" . ".join)) if len(cdocs)
              else sparse.csr_matrix((0, F)))
        links = []
        for lag in range(LAG_DAYS, WINDOW_DAYS + 1):
            day = rows.assign(doc_date=rows.date - pd.Timedelta(days=lag))
            mem = day[day.ticker != INDEX].merge(cdocs, left_on=["ticker", "doc_date"],
                                                 right_on=["ticker", "date"], suffixes=("", "_doc"))
            links.append(mem.assign(wt=1.0)[["i", "j", "wt", "count"]])
            idx = day[day.ticker == INDEX]
            if len(idx):   # headlines tagged with the index itself (a single-stock run's own news)
                own = idx.merge(cdocs[cdocs.ticker == INDEX], left_on=["ticker", "doc_date"],
                                right_on=["ticker", "date"], suffixes=("", "_doc"))
                links.append(own.assign(wt=1.0)[["i", "j", "wt", "count"]])
            if len(idx):   # the index reads every member's news, weighted by its Dow weight on day t
                m = idx.merge(cdocs.rename(columns={"ticker": "member"}), left_on="doc_date",
                              right_on="date", suffixes=("", "_doc"))
                w = self._index_weights(idx.date, ctx).rename(columns={"ticker": "member"})
                m = m.merge(w, on=["date", "member"])
                links.append(m.rename(columns={"w": "wt"})[["i", "j", "wt", "count"]])
        lk = pd.concat(links, ignore_index=True) if links else pd.DataFrame(columns=["i", "j", "wt", "count"])
        A = sparse.csr_matrix((lk.wt.astype(float), (lk.i.astype(int), lk.j.astype(int))), shape=(n, len(cdocs)))
        Xc = normalize(A @ Vc)
        n_company = np.bincount(lk.i.astype(int), weights=lk["count"].astype(float), minlength=n)
        # newest company doc per row, for showing example headlines
        newest = lk.merge(cdocs[["j", "date", "headline"]], on="j").sort_values(["i", "date", "wt"])
        c_sample = newest.groupby("i").headline.last().map(lambda h: " | ".join(h[:3]))

        # ---- market block: the most recent market-news day in the window
        mdays = mk.groupby("date").headline.agg(list).sort_index()
        Vm = self.vectorizer.transform(mdays.map(" . ".join)) if len(mdays) else sparse.csr_matrix((0, F))
        mday_index = mdays.index.values
        pos = np.searchsorted(mday_index, (rows.date - pd.Timedelta(days=LAG_DAYS)).values, side="right") - 1
        ok = pos >= 0
        ok[ok] = mday_index[pos[ok]] >= (rows.date - pd.Timedelta(days=WINDOW_DAYS)).values[ok]
        sel = sparse.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), pos[ok])), shape=(n, len(mdays)))
        Xm = normalize(sel @ Vm)
        n_market = np.zeros(n)
        n_market[ok] = mdays.map(len).values[pos[ok]]
        m_sample = pd.Series(mdays.map(lambda h: " | ".join(h[:2])).values[pos[ok]], index=np.flatnonzero(ok))

        X = sparse.hstack([Xc, Xm]).tocsr()
        is_index = (rows.ticker == INDEX).to_numpy()
        # retrieval vectors: members compare their own company news; the index compares everything
        R = normalize(sparse.hstack([Xc, sparse.diags(is_index.astype(float)) @ Xm]).tocsr())
        # example headlines to show: the company's own news for members, world headlines for the index
        cs, ms = c_sample.reindex(range(n)), m_sample.reindex(range(n))
        sample = cs.fillna(ms).where(~is_index, ms.fillna(cs)).fillna("")
        return {"X": X, "R": R, "n_company": n_company, "n_market": n_market,
                "sample": sample.to_numpy(), "rows": rows}

    # ================================================================== similar past days
    def _car(self, ctx: Context) -> pd.DataFrame:
        """5-day market-adjusted return after each day, from past-only alpha and beta."""
        px = ctx.prices[["date", "ticker", "adj_close"]].sort_values("date")
        mkt = px[px.ticker == INDEX].set_index("date").adj_close
        rm = np.log(mkt).diff()
        out = []
        for t, g in px.groupby("ticker"):
            r = np.log(g.set_index("date").adj_close).diff()
            if t == INDEX:
                ar = r
            else:
                m = rm.reindex(r.index)
                w = r.rolling(BETA_DAYS, min_periods=120)
                beta = w.cov(m) / m.rolling(BETA_DAYS, min_periods=120).var()
                alpha = w.mean() - beta * m.rolling(BETA_DAYS, min_periods=120).mean()
                beta, alpha = beta.shift(1), alpha.shift(1)     # fitted on days before t only
                ar = r - (alpha + beta * m)
            car = ar.rolling(CAR_DAYS).sum().shift(-CAR_DAYS)    # sum of AR over t+1 .. t+5
            out.append(pd.DataFrame({"date": car.index, "ticker": t, "car": car.values}))
        return pd.concat(out, ignore_index=True)

    def _retrieve(self, R: sparse.csr_matrix, rows: pd.DataFrame, k: int = TOP_K, return_top: int = 0):
        """For each query row: similarity-weighted average CAR of its k most similar past library days
        (same ticker, outcome known by day t). Optionally also returns the top matches."""
        out = np.full(len(rows), np.nan)
        tops: list = [[] for _ in range(len(rows))]
        for ticker, qi in rows.groupby("ticker").indices.items():
            lib = self.library.get(ticker)
            if lib is None:
                continue
            for start in range(0, len(qi), 2000):     # chunks keep memory small
                q = qi[start:start + 2000]
                S = (R[q] @ lib["R"].T).toarray()
                t = rows.date.values[q].astype("datetime64[ns]")
                S[lib["label_known"][None, :] > t[:, None]] = 0.0     # outcome not known yet -> unusable
                kk = min(k, S.shape[1])
                if kk == 0:
                    continue
                best = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
                sims = np.take_along_axis(S, best, axis=1)
                tot = sims.sum(1)
                val = (sims * lib["car"][best]).sum(1) / np.where(tot > 0, tot, 1)
                out[q] = np.where(tot > 0, val, np.nan)
                if return_top:
                    for r_, b, s in zip(q, best, sims):
                        order = np.argsort(-s)[:return_top]
                        tops[r_] = [(int(b[o]), float(s[o])) for o in order if s[o] > 0]
        return (out, tops) if return_top else out

    # ================================================================== fitting
    def _volume_z(self, rows: pd.DataFrame, n_company: np.ndarray) -> np.ndarray:
        v = np.log1p(n_company)
        mu = rows.ticker.map(lambda t: self.vol_stats.get(t, self.vol_stats["_all"])[0]).to_numpy()
        sd = rows.ticker.map(lambda t: self.vol_stats.get(t, self.vol_stats["_all"])[1]).to_numpy()
        return (v - mu) / sd

    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        df = panel.dropna(subset=["y"]).reset_index(drop=True)
        self._learn_payoffs(df)
        is_idx = df.ticker == INDEX
        self.base_rate = {"index": float(df.y[is_idx].mean()), "member": float(df.y[~is_idx].mean())}

        # 1. vocabulary from training headlines only (dated up to the last fit row)
        last = df.date.max()
        heads = []
        if ctx.market_news is not None:
            heads.append(ctx.market_news.loc[ctx.market_news.date <= last, "headline"])
        if ctx.company_news is not None:
            cn = ctx.company_news
            heads.append(cn.loc[(cn.date <= last) & cn.ticker.isin(set(df.ticker)), "headline"])
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                                          stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                                          dtype=np.float32)
        self.vectorizer.fit(pd.concat(heads).drop_duplicates())
        self.vectorizer.stop_words_ = None   # a big list of dropped words we don't need to save
        F = len(self.vectorizer.vocabulary_)

        # 2. text vectors for every row; keep the rows that have any news
        V = self._vectors(df, ctx)
        has = (V["n_company"] + V["n_market"]) > 0
        news = df[has].reset_index(drop=True)
        X, R = V["X"][has], V["R"][has]
        print(f"  news_reaction: {has.sum():,} of {len(df):,} fit rows have news "
              f"({news.date.min().date() if len(news) else '-'} to {news.date.max().date() if len(news) else '-'}), "
              f"vocabulary {F:,}")

        # 3. library of past days for retrieval (per ticker), with their 5-day abnormal returns
        car = self._car(ctx)
        news = news.merge(car, on=["date", "ticker"], how="left")
        self.library = {}
        for t, ii in news.groupby("ticker").indices.items():
            keep = ii[(R[ii].getnnz(axis=1) > 0) & news.car.notna().values[ii]]
            if len(keep):
                self.library[t] = {"R": R[keep], "car": news.car.values[keep],
                                   "fwd_return": news.fwd_return.values[keep],
                                   "label_known": news.label_known.values[keep].astype("datetime64[ns]"),
                                   "date": news.date.values[keep], "sample": V["sample"][has][keep]}
        retrieved = np.nan_to_num(self._retrieve(R, news[["date", "ticker"]]), nan=0.0)

        # 4. volume statistics (log headline count), per ticker
        v = np.log1p(V["n_company"][has])
        self.vol_stats = {"_all": (float(v.mean()), float(v.std() or 1.0))}
        for t, ii in news.groupby("ticker").indices.items():
            self.vol_stats[t] = (float(v[ii].mean()), float(v[ii].std() or 1.0) or 1.0)
        vol_z = self._volume_z(news, V["n_company"][has])

        # 5. out-of-fold p_text, time ordered: train on fold 1 -> predict fold 2; folds 1-2 -> fold 3.
        #    Training rows whose label isn't known before the predicted fold starts are dropped.
        y = news.y.astype(int).to_numpy()
        days = np.sort(news.date.unique())
        cuts = [days[0], days[len(days) // 3], days[2 * len(days) // 3]]
        oof = np.full(len(news), np.nan)

        def text_lr(C):
            return LogisticRegression(C=C, solver="liblinear", max_iter=200)

        # pick C on fold 3 (inside the fit years only)
        tr = (news.label_known < cuts[2]).to_numpy()
        te = (news.date >= cuts[2]).to_numpy()
        scores = {C: log_loss(y[te], text_lr(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], labels=[0, 1])
                  for C in C_GRID}
        self.C = min(scores, key=scores.get)
        base_ll = log_loss(y[te], np.full(te.sum(), y[tr].mean()), labels=[0, 1])
        print(f"  news_reaction: fold-3 log loss by C {({c: round(v, 4) for c, v in scores.items()})}, "
              f"base rate alone {base_ll:.4f}")
        for k in (1, 2):
            tr = (news.label_known < cuts[k]).to_numpy()
            te = ((news.date >= cuts[k]) & (news.date < (cuts[k + 1] if k + 1 < 3 else days[-1] + 1))).to_numpy()
            oof[te] = text_lr(self.C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

        # 6. the combiner, fitted on out-of-fold rows only
        ok = ~np.isnan(oof)
        Z = np.column_stack([_logit(oof), retrieved, vol_z])
        self.combiner = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
        self.combiner.fit(Z[ok], y[ok])
        self.oof_text_accuracy = float(((oof[ok] >= 0.5) == y[ok]).mean())

        # 7. final text model on every row with news
        self.text_model = text_lr(self.C).fit(X, y)
        lr = self.combiner.steps[1][1]
        print(f"  news_reaction: C={self.C}, out-of-fold text accuracy {self.oof_text_accuracy:.3f}, "
              f"combiner weights (logit p_text, retrieved, volume) = {np.round(lr.coef_[0], 3).tolist()}")

    # ================================================================== predicting
    def details(self, panel: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        """Everything the analyst computes for each row, including the final probability `p`."""
        rows = panel[["date", "ticker"]].reset_index(drop=True)
        V = self._vectors(rows, ctx)
        n_all = V["n_company"] + V["n_market"]
        has = n_all > 0
        out = pd.DataFrame({
            "has_news": has.astype(float), "n_headlines": n_all, "n_company_headlines": V["n_company"],
            "n_market_headlines": V["n_market"], "p_text": np.nan, "retrieved_return": np.nan,
            "news_volume_z": self._volume_z(rows, V["n_company"]),
        })
        base = np.where(rows.ticker == INDEX, self.base_rate["index"], self.base_rate["member"])
        out["p"] = base
        if has.any():
            p_text = self.text_model.predict_proba(V["X"][has])[:, 1]
            retrieved = self._retrieve(V["R"][has], rows[has].reset_index(drop=True))
            Z = np.column_stack([_logit(p_text), np.nan_to_num(retrieved, nan=0.0), out.news_volume_z[has]])
            out.loc[has, "p_text"] = p_text
            out.loc[has, "retrieved_return"] = retrieved
            out.loc[has, "p"] = self.combiner.predict_proba(Z)[:, 1]
        self.last_coverage = float(has.mean()) if len(has) else 0.0
        out.index = panel.index
        return out

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        return self.details(panel, ctx)["p"].to_numpy()

    def key_numbers(self, row: pd.Series, ctx: Context) -> dict[str, float]:
        d = self.details(row.to_frame().T.assign(date=pd.Timestamp(row["date"])), ctx).iloc[0]
        return {k: float(d[k]) for k in self.features if pd.notna(d[k])}

    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        """The words in this row's headlines that pushed p_text up or down the most."""
        one = pd.DataFrame({"date": [pd.Timestamp(row["date"])], "ticker": [row["ticker"]]})
        x = self._vectors(one, ctx)["X"]
        contrib = x.multiply(self.text_model.coef_[0]).tocsr()
        names = self.word_names()
        order = np.argsort(-np.abs(contrib.data))[:5]
        return [(names[contrib.indices[o]], float(contrib.data[o])) for o in order]

    def word_names(self) -> np.ndarray:
        words = self.vectorizer.get_feature_names_out()
        return np.concatenate([["company: " + w for w in words], ["market: " + w for w in words]])

    def top_words(self, n: int = 15) -> dict[str, list[tuple[str, float]]]:
        """The learned 'good news' and 'bad news' words (largest positive / negative weights)."""
        w = self.text_model.coef_[0]
        names = self.word_names()
        up, down = np.argsort(-w)[:n], np.argsort(w)[:n]
        return {"positive": [(names[i], round(float(w[i]), 3)) for i in up],
                "negative": [(names[i], round(float(w[i]), 3)) for i in down]}

    def evidence(self, ticker: str, date, ctx: Context, n: int = 5) -> list[dict]:
        """For the report: the n most similar past news days for this ticker and what happened next."""
        one = pd.DataFrame({"date": [pd.Timestamp(date)], "ticker": [ticker]})
        V = self._vectors(one, ctx)
        _, tops = self._retrieve(V["R"], one, return_top=n)
        lib = self.library.get(ticker, {})
        return [{"date": str(pd.Timestamp(lib["date"][j]).date()), "similarity": round(s, 3),
                 "headlines": str(lib["sample"][j]),
                 "abnormal_return_5d": round(float(lib["car"][j]), 4),
                 "return_20d": round(float(lib["fwd_return"][j]), 4)} for j, s in tops[0]]


def build() -> NewsReaction:
    return NewsReaction()

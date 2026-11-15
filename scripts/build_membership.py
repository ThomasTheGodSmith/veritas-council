"""Build data/raw/membership/dow_membership.csv: who was in the Dow, and when (PLAN.md §0b).

Source: Wikipedia, "Historical components of the Dow Jones Industrial Average"
(the wikitext is saved next to the output). The change events below are copied from its
"Summary of changes since 1991". Tickers are mapped by hand.

Output columns:
  member      stable id for the company while it was in the Dow
  company     name while in the Dow
  yf_ticker   Yahoo Finance ticker whose price history continues this company, or blank
              when no price history is available (delisted / acquired / bankrupt)
  start_date  first trading day as a member (changes take effect at the open)
  end_date    first trading day NOT a member (exclusive); blank = still a member
  note        caveats about the price history

Run:  uv run python scripts/build_membership.py
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

OUT = Path("data/raw/membership/dow_membership.csv")
START = date(1991, 5, 6)  # snapshot below is the index as of this date

# member id -> (company, yf_ticker or "", note)
COMPANIES: dict[str, tuple[str, str, str]] = {
    "ALD": ("Allied-Signal / Honeywell International", "HON", ""),
    "AA": ("Aluminum Company of America / Alcoa Inc.", "HWM", "Old Alcoa Inc. continues as Howmet Aerospace (HWM) after the 2016 split and 2020 spin-off; check history depth"),
    "AXP": ("American Express", "AXP", ""),
    "T_OLD": ("AT&T Corporation (old)", "", "Acquired by SBC in 2005; no price history on Yahoo"),
    "BS": ("Bethlehem Steel", "", "Bankrupt 2001; no price history"),
    "BA": ("Boeing", "BA", ""),
    "CAT": ("Caterpillar", "CAT", ""),
    "CVX": ("Chevron", "CVX", ""),
    "KO": ("Coca-Cola", "KO", ""),
    "DD": ("E.I. du Pont de Nemours / DowDuPont", "DD", "DD ticker line: old DuPont -> DowDuPont (2017) -> DuPont de Nemours (2019). Spin-offs not adjusted"),
    "EK": ("Eastman Kodak", "", "Bankrupt 2012; old shares cancelled, no price history"),
    "XOM": ("Exxon / ExxonMobil", "XOM", ""),
    "GE": ("General Electric", "GE", "Several spin-offs (2023 GEHC, 2024 GEV) not split-adjusted"),
    "GM": ("General Motors (old)", "", "Bankrupt 2009 (Motors Liquidation); no price history"),
    "GT": ("Goodyear", "GT", ""),
    "IBM": ("IBM", "IBM", ""),
    "IP": ("International Paper", "IP", ""),
    "JPM": ("J.P. Morgan & Co. / JPMorgan Chase", "JPM", "Before the 2000 merger, Yahoo's JPM history is Chase Manhattan, not J.P. Morgan & Co."),
    "MCD": ("McDonald's", "MCD", ""),
    "MRK": ("Merck", "MRK", ""),
    "MMM": ("3M", "MMM", ""),
    "MO": ("Philip Morris / Altria", "MO", "Kraft (2007) and PMI (2008) spin-offs not split-adjusted"),
    "PG": ("Procter & Gamble", "PG", ""),
    "S": ("Sears Roebuck", "", "Acquired by Kmart 2005; no price history"),
    "TX": ("Texaco", "", "Acquired by Chevron 2001; no price history"),
    "UK": ("Union Carbide", "", "Acquired by Dow Chemical 2001; no price history"),
    "UTX": ("United Technologies / Raytheon Technologies", "RTX", "RTX is the legal successor of UTX"),
    "DIS": ("Walt Disney", "DIS", ""),
    "WX": ("Westinghouse Electric", "", "Became CBS 1997; no continuous price history"),
    "Z": ("F. W. Woolworth", "FL", "Woolworth -> Venator -> Foot Locker (FL); check availability"),
    "TRV_OLD": ("Travelers Inc. / Citigroup", "C", "Travelers Group merged with Citicorp to form Citigroup (1998); 1:10 reverse split 2011"),
    "HPQ": ("Hewlett-Packard", "HPQ", ""),
    "JNJ": ("Johnson & Johnson", "JNJ", ""),
    "WMT": ("Walmart", "WMT", ""),
    "MSFT": ("Microsoft", "MSFT", ""),
    "INTC": ("Intel", "INTC", ""),
    "SBC": ("SBC Communications / AT&T Inc.", "T", "SBC renamed AT&T Inc. on 2005-11-21; same ticker line"),
    "HD": ("Home Depot", "HD", ""),
    "AIG": ("American International Group", "AIG", "1:20 reverse split 2009"),
    "PFE": ("Pfizer", "PFE", ""),
    "VZ": ("Verizon", "VZ", ""),
    "BAC": ("Bank of America", "BAC", ""),
    "KFT": ("Kraft Foods Inc.", "MDLZ", "Kraft Foods Inc. renamed Mondelez (MDLZ) after the 2012 split"),
    "TRV": ("The Travelers Companies", "TRV", ""),
    "CSCO": ("Cisco", "CSCO", ""),
    "UNH": ("UnitedHealth", "UNH", ""),
    "GS": ("Goldman Sachs", "GS", ""),
    "NKE": ("Nike", "NKE", ""),
    "V": ("Visa", "V", ""),
    "AAPL": ("Apple", "AAPL", ""),
    "WBA": ("Walgreens Boots Alliance", "WBA", "Taken private in 2025; check availability"),
    "DOW": ("Dow Inc.", "DOW", ""),
    "HON": ("Honeywell", "HON", ""),
    "AMGN": ("Amgen", "AMGN", ""),
    "CRM": ("Salesforce", "CRM", ""),
    "AMZN": ("Amazon", "AMZN", ""),
    "NVDA": ("Nvidia", "NVDA", ""),
    "SHW": ("Sherwin-Williams", "SHW", ""),
    "GOOGL": ("Alphabet", "GOOGL", "Class A shares assumed"),
}

MEMBERS_1991 = [
    "ALD", "AA", "AXP", "T_OLD", "BS", "BA", "CAT", "CVX", "KO", "DD",
    "EK", "XOM", "GE", "GM", "GT", "IBM", "IP", "JPM", "MCD", "MRK",
    "MMM", "MO", "PG", "S", "TX", "UK", "UTX", "DIS", "WX", "Z",
]

# (effective date, added, removed). Name-only changes (e.g. 2003 Altria, 2005 AT&T Inc.,
# 2017 DowDuPont when it keeps the DD line) are folded into the member id.
CHANGES: list[tuple[date, list[str], list[str]]] = [
    (date(1997, 3, 17), ["TRV_OLD", "HPQ", "JNJ", "WMT"], ["WX", "TX", "BS", "Z"]),
    (date(1999, 11, 1), ["MSFT", "INTC", "SBC", "HD"], ["GT", "S", "UK", "CVX"]),
    (date(2004, 4, 8), ["AIG", "PFE", "VZ"], ["T_OLD", "EK", "IP"]),
    (date(2008, 2, 19), ["CVX", "BAC"], ["MO", "ALD"]),
    (date(2008, 9, 22), ["KFT"], ["AIG"]),
    (date(2009, 6, 8), ["TRV", "CSCO"], ["GM", "TRV_OLD"]),
    (date(2012, 9, 24), ["UNH"], ["KFT"]),
    (date(2013, 9, 23), ["GS", "NKE", "V"], ["AA", "BAC", "HPQ"]),
    (date(2015, 3, 19), ["AAPL"], ["SBC"]),
    # 2017-09-01 DowDuPont replaced DuPont: same DD ticker line, kept as one member.
    (date(2018, 6, 26), ["WBA"], ["GE"]),
    (date(2019, 4, 2), ["DOW"], ["DD"]),
    # 2020-04-06 Raytheon Technologies replaced United Technologies: same RTX line.
    (date(2020, 8, 31), ["AMGN", "HON", "CRM"], ["XOM", "PFE", "UTX"]),
    (date(2024, 2, 26), ["AMZN"], ["WBA"]),
    (date(2024, 11, 8), ["NVDA", "SHW"], ["INTC", "DOW"]),
    (date(2026, 6, 29), ["GOOGL"], ["VZ"]),
]

# Current members per the Wikipedia June 29, 2026 snapshot, as member ids.
EXPECTED_2026 = {
    "MMM", "KO", "NKE", "GOOGL", "GS", "NVDA", "AMZN", "HD", "PG", "AXP", "HON", "CRM",
    "AMGN", "IBM", "SHW", "AAPL", "JNJ", "TRV", "BA", "JPM", "UNH", "CAT", "MCD", "V",
    "CVX", "MRK", "WMT", "CSCO", "MSFT", "DIS",
}


def build() -> list[dict[str, str]]:
    current: dict[str, date] = {m: START for m in MEMBERS_1991}
    rows: list[dict[str, str]] = []
    for when, added, removed in CHANGES:
        assert len(added) == len(removed), when
        for m in removed:
            assert m in current, f"{when}: {m} removed but not a member"
            rows.append(_row(m, current.pop(m), when))
        for m in added:
            assert m not in current, f"{when}: {m} added twice"
            current[m] = when
        assert len(current) == 30, f"{when}: {len(current)} members"
    assert set(current) == EXPECTED_2026, set(current) ^ EXPECTED_2026
    rows.extend(_row(m, start, None) for m, start in current.items())
    return sorted(rows, key=lambda r: (r["start_date"], r["member"]))


def _row(member: str, start: date, end: date | None) -> dict[str, str]:
    company, ticker, note = COMPANIES[member]
    return {
        "member": member,
        "company": company,
        "yf_ticker": ticker,
        "start_date": start.isoformat(),
        "end_date": end.isoformat() if end else "",
        "note": note,
    }


if __name__ == "__main__":
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} membership spells to {OUT}")

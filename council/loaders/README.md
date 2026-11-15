# Loaders

One file per dataset provided. Each loader returns the standard format:

- Prices: `date, ticker, open, high, low, close, volume`
- News: `date, ticker (or "DJIA"), headline, source`
- Membership: `ticker, date_added, date_removed`
- Splits: `date, ticker, ratio`
- Macro: `date, series, value`

Swapping datasets means changing a loader and `config/data.yaml`, never the analysts.

# Results & Findings

This document records what the LSTM forecaster actually achieves against
rigorous baselines, and walks through two engineering questions that were
tested empirically rather than assumed: an earnings-proximity feature
(implemented, tested, and **rejected**), and how much historical data is
worth training on (tested across 2/5/10 years — answer: it depends on the
ticker).

## Methodology

**Backtest design** (`scripts/backtest.py`): walk-forward evaluation on
held-out real market data (via `yfinance`). At each point in the held-out
window, three forecasters independently predict the next 7 trading days:

- **Naive** — flat line at the last known price
- **Drift** — extrapolates the average daily log-return over the trailing 60 days
- **LSTM** — the trained model, predicting a *residual* over the drift
  baseline (see "Why residual-over-drift" below)

Each is scored against what actually happened, on two metrics:
- **MAE** (mean absolute error, in dollars) — how far off the price forecast was
- **Directional accuracy** — did the forecast correctly call up vs. down over
  the 7-day horizon (arguably the more decision-relevant metric for a
  buy/sell/hold tool)

To check whether results are a stable property of the model or an artifact
of one lucky/unlucky stretch, the held-out period is also split into
**rolling sub-windows** (e.g. four 30-day chunks inside a 120-day test
period), each scored independently.

**Training determinism**: all comparisons below use a fixed random seed
(`torch.manual_seed(42)` / `np.random.seed(42)`) set immediately before model
initialization, so that differences between runs reflect actual changes to
the feature set or architecture — not random weight-initialization luck.
Early runs of this project (visible in project history) did **not** control
for this, and produced misleading before/after comparisons as a result —
worth flagging as a lesson in its own right: an ML ablation without a fixed
seed is not a controlled experiment.

## Why residual-over-drift, not raw returns

Early versions of this model predicted the horizon's cumulative log-return
directly. Since the drift baseline (a simple historical-average
extrapolation) already captures most of what's predictable about a stock's
short-term trend, an LSTM predicting raw returns spends most of its capacity
re-learning what a two-line drift calculation gives for free. Switching the
training target to the **residual over drift** — actual return minus drift's
predicted return — forces the model to focus on the part of the forecast
drift *doesn't* explain, and makes it directly comparable to the drift
baseline at inference time. Implementation: `app/models/baselines.py`,
`app/models/train.py::build_sequences()`, `app/models/predict.py`.

## Baseline comparison (6 core features)

6 features: `close, volume, sma_10, sma_30, rsi_14, macd`. 120-day held-out
period, split into four 30-day rolling windows, real market data via
`yfinance`.

| Ticker | Overall MAE vs. best baseline | Overall directional accuracy | Windows won (of 4) |
|---|---|---|---|
| AMZN | 6.1% worse | 60.0% | 1 |
| NVDA | 0.1% *better* | 55.8% | 1 |
| MSFT | 7.1% *better* | 52.5% | — |
| GOOGL | 12.8% worse | 40.0% | — |
| AAPL | 12.9% worse | 30.0% | — |

**Takeaway**: the LSTM is not a consistent win. It roughly ties or narrowly
beats baselines on 2 of 5 tickers (MSFT, NVDA) and loses on the other 3. This
is a realistic and, frankly, expected outcome for short-horizon (7-day)
single-stock forecasting — efficient-markets intuition predicts that a
model conditioned only on price history should struggle to reliably beat a
"no change" or "recent trend" baseline, since near-term price moves are
dominated by information the model can't see (news, earnings surprises,
macro shocks). A model that swept all five tickers would actually be a
yellow flag for small-sample luck, not a stronger result.

**Directional accuracy sits close to a coin flip across the board** (30–60%
range). This is the honest caveat for the buy/sell/hold framing: the model's
qualitative "hold" or "buy" call should be read as a probabilistic lean, not
a confident prediction, and the GenAI synthesis layer's job (see below) is to
combine this admittedly noisy quantitative signal with qualitative context a
pure price-history model has no access to.

## Case study: AMZN, earnings volatility, and a rejected feature

### The observation
A deeper look at AMZN's rolling windows showed one window (2026-03-30 to
2026-05-11) where the LSTM's price-magnitude error spiked sharply — MAE
$15.7 vs. naive's $9.9, a ~58% gap — while its *directional* accuracy in that
same window was actually strong (76.7%). Cross-referencing against
AMZN's actual earnings calendar confirmed this window contained Amazon's
Q1 2026 earnings report (April 29, 2026), a scheduled event with elevated
implied volatility (~43%) going in.

### The hypothesis
A pure price-history LSTM has no way to "see" a scheduled earnings report
coming — it can only react to volatility after the fact. Giving the model an
explicit feature marking "an earnings report falls within this forecast
horizon" should let it learn to widen or dampen its point estimate around
these known events, rather than being blindsided.

### The implementation
`app/data/earnings_calendar.py` — fetches real historical/upcoming earnings
dates via `yfinance`, with a synthetic-quarterly-schedule fallback for
offline use. Adds two features: `days_to_earnings` (signed distance to the
nearest report) and `earnings_in_horizon` (binary: does an earnings date
fall within the model's 7-day forecast window). Wired into
`market_data.py::get_prepared_data()` and added to `config.FEATURES`.

### The controlled test
Retrained AMZN with a fixed seed, once with 6 features and once with 8
(adding the earnings features), all else identical:

| | 6 features | 8 features (+ earnings) |
|---|---|---|
| Overall MAE vs. best baseline | 6.1% worse | 20.2% worse |
| Overall directional accuracy | **60.0%** | 52.5% |
| Earnings-quarter window (Window 3) MAE vs. best baseline | 23.4% worse | **59.2% worse** |
| Earnings-quarter window directional accuracy | 43.3% | 20.0% |

### The result: the hypothesis was rejected
The earnings feature made performance **worse overall, and worse
specifically in the earnings-quarter window it was designed to help** — the
opposite of the intended effect. A follow-up (partially confounded — see
note below) test on NVDA was directionally consistent with this: no benefit
observed.

### Why, most likely
With ~370 training sequences per ticker and `earnings_in_horizon` true for
only ~8% of rows, there is probably too little data for the LSTM to learn a
reliable earnings-specific adjustment. The two extra input dimensions likely
added representational noise that the optimizer partially overfit to in the
92% "normal" regime, at the cost of the 8% "earnings" regime they were meant
to help — the opposite of the intended trade-off.

### A secondary, unplanned finding
While testing this on NVDA, `yfinance.get_earnings_dates()` failed with a
`KeyError` distinct from the `lxml`-related failures seen on other tickers
earlier in development — confirming this API's schema is fragile/inconsistent
across tickers or calls. This validates the decision to build a graceful
synthetic-fallback path into `earnings_calendar.py` rather than assuming the
real data source is always reliably reachable in the expected shape — the
same defensive pattern already used in `market_data.py` and `news_corpus.py`
for their respective external dependencies.

### Decision
The earnings-proximity features are **not** included in the shipped feature
set (`config.FEATURES` reverted to the 6-feature version). The
`earnings_calendar.py` module is retained in the codebase, documented here
as a tested-and-rejected hypothesis rather than deleted — the ablation
methodology and the fallback-handling pattern are both reusable even though
this particular feature didn't pan out.

## Case study: how much historical data to train on

### The question
The earnings-feature ablation above pointed at a likely root cause:
~370 training sequences per ticker (from ~2 years of daily data) may simply
be too little data for the model to learn reliable patterns from, especially
for anything with a low base rate. The obvious follow-up: does giving the
model more historical data to train on actually help?

### The test
Same fixed-seed methodology as above, same 6-feature set, same 120-day
held-out backtest period — the only variable changed is how much history
(`yfinance` period → training-sequence count) each model is trained on.
Compared 2 years (~370 sequences), 5 years (~950 sequences), and 10 years
(~1,970 sequences) of daily data per ticker.

**Overall MAE vs. best baseline, by data volume (positive = LSTM wins):**

| Ticker | 2yr | 5yr | 10yr | Pattern |
|---|---|---|---|---|
| AAPL | -12.9% | -3.3% | **-1.6%** | Monotonic improvement with more data |
| MSFT | **+7.1%** | +2.7% | +0.6% | Monotonic *decline* with more data |
| GOOGL | -12.8% | -10.4% | -8.7% | Loses at every data volume tested |
| NVDA | +0.1% | -2.6% | **+2.1%** | Non-monotonic, best at 10yr |
| AMZN | **-6.1%** | -6.8% | -9.1% | Monotonic decline with more data |

### The result: there is no universal answer, and that itself is the finding
AAPL and MSFT move in **opposite directions** as training history increases
— ruling out a simple "more data is always better" (or always worse) story.
The right amount of training history appears to be **ticker-specific**,
plausibly tied to how stable vs. regime-shifted each individual stock's own
multi-year price history has been (e.g. AMZN's 2020-2022 hypergrowth-and-
correction cycle and 2022 stock split plausibly introduce old-regime noise
that a longer lookback window can't fully separate from current behavior).

**GOOGL is a separate, more interesting finding**: it underperforms both
baselines at *every* data volume tested — 2yr, 5yr, and 10yr alike, including
losing in 4/4 rolling sub-windows at 10yr ("consistently underperforming,"
not a lucky/unlucky stretch). Since more or less history doesn't move this
result at all, the problem for GOOGL specifically is not a data-volume issue
— it's something else (feature set, hyperparameters, or simply a stock whose
short-term dynamics this architecture doesn't capture well) — and is flagged
below as a known, open limitation rather than something more historical data
will fix.

### Decision
**10 years** was selected as the shipped default (`fetch_ohlcv(period="10y")`,
`train.py` fetch window sized accordingly). Rationale: best-or-near-best on
3 of 5 tickers (AAPL, NVDA outright; MSFT narrowly positive), and "use all
reasonably available history" is a simpler, more defensible default than a
per-ticker-tuned choice that risks looking cherry-picked. GOOGL's underlying
issue is unaffected by this choice either way.

## What would be worth trying next, if extending this further

- **A coarser earnings signal.** Day-level precision (`days_to_earnings`)
  may be harder to learn than a simple categorical "earnings week / not
  earnings week" flag with less granularity to overfit around — worth
  revisiting now that 10 years of history gives the model roughly 5x more
  earnings-adjacent examples than the original ablation had access to.
- **Investigate GOOGL specifically.** Since more/less training history
  didn't change its outcome, the next lever to pull is architecture or
  feature set (different `SEQ_LEN`, additional features, or simply accepting
  this ticker as an out-of-scope limitation for this iteration) rather than
  more data-volume sweeps.
- **Separate volatility-scale modeling.** Rather than folding earnings
  proximity into the same residual-magnitude prediction, a two-headed model
  (predict direction and predict a separate volatility/uncertainty scale)
  might isolate the "I don't know how big this move will be" signal more
  cleanly than asking one regression head to do both jobs at once.
- **Ensemble the GenAI layer's awareness of earnings instead.** The RAG/news
  pipeline is well-positioned to catch "AMZN reports earnings tomorrow" from
  real news text — arguably a more reliable signal source for this specific
  regime than trying to teach a small LSTM to infer it from price history
  alone.

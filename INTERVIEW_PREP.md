# Interview Prep — AI Financial Analyst Project

Organized by category. Each answer is grounded in what you actually built and
found — practice saying these in your own words rather than memorizing verbatim.

---

## 1. The elevator pitch (have this cold, 30 seconds)

**Q: Walk me through this project.**

"I built a system that combines two things: an LSTM neural network that
forecasts a stock's price 7 days out from historical price data, and a
separate RAG-powered pipeline that retrieves real news and uses an LLM to
generate a buy/sell/hold narrative. But the more important part is what I did
after building it — I backtested the model rigorously against naive
baselines, found it only reliably beat them on 2 of 5 tickers, diagnosed a
specific failure mode tied to an earnings event, built a fix, tested it with
a controlled experiment, and found the fix actually hurt performance — so I
documented that and didn't ship it. The project is as much about the
evaluation discipline as the model itself."

---

## 2. Deep learning / LSTM questions

**Q: Why did you use an LSTM instead of a simpler model or a Transformer?**

LSTMs are built for sequential data — they maintain an internal memory state
as they read through the 30-day lookback window, letting the model learn
temporal patterns (e.g., sustained trend vs. single-day spike) that a
feedforward network can't represent at all, since it would see 30 days as an
unordered bag of numbers. I considered a Temporal Fusion Transformer (my
original spec named it as an alternative) but chose LSTM for scope reasons —
TFT is more powerful but has significantly more implementation complexity
(attention mechanisms, separate static/known/observed input handling) that
wasn't justified given my actual bottleneck turned out to be data volume and
predictability, not architecture (see backtest findings below).

**Q: What does your model actually predict — raw price?**

No — it predicts a *residual over a drift baseline*. The drift baseline is a
simple extrapolation of the average recent daily return. Since that baseline
already captures most of what's predictable about a trending stock, I
trained the LSTM to predict only the part the drift baseline doesn't already
explain. This makes the model's job harder in a useful way — it can't just
relearn "prices tend to keep trending," it has to find something beyond that.

**Q: How do you get a confidence interval from a single model?**

Monte Carlo dropout. At inference, I run the model 30 times with dropout
layers intentionally left active (normally dropout is disabled at inference).
Each pass randomly zeroes out some internal connections, producing a
slightly different prediction each time. The spread of those 30 predictions
becomes the 10th–90th percentile confidence band; the median becomes the
point forecast. It's a cheap approximation of a predictive distribution
without training a full ensemble of separate models.

**Q: Why residual-over-drift instead of predicting price directly?**

Predicting raw price (or even raw returns) means the model spends capacity
re-learning what a two-line drift calculation already gives for free.
Predicting the residual forces the LSTM to focus specifically on the part
that's genuinely hard to predict, and makes it directly comparable to the
drift baseline at evaluation time — same target space, apples to apples.

**Q: What features does the model use?**

Six: close price, volume, two moving averages (10-day and 30-day), RSI
(momentum/overbought-oversold), and MACD (trend strength). All derived from
raw OHLCV data via `yfinance`.

**Q: Did you try adding more features?**

Yes — I built an earnings-proximity feature (days until the next earnings
report, and a binary flag for whether an earnings date falls within the
7-day forecast horizon), hypothesizing it would help the model anticipate
volatility around scheduled events. I tested it with a controlled ablation
(same random seed, same data, only the feature set changed) and found it
made performance *worse* — both overall and specifically in the
earnings-quarter window it was meant to help. My best explanation: with only
~370-1,970 training sequences per ticker and the earnings flag true for only
a small fraction of rows, there wasn't enough data for the model to learn a
reliable earnings-specific adjustment — the extra dimensions likely added
noise rather than signal. I documented this as a rejected hypothesis rather
than quietly dropping it.

---

## 3. Backtesting / rigor questions (your strongest material — lean into these)

**Q: How do you know your model is any good?**

I built a walk-forward backtest that replays history: at many past points,
it generates a 7-day forecast using the LSTM and two baselines (naive — "no
change," and drift — "extrapolate recent trend"), then checks what actually
happened. Across 5 tickers over a 120-day held-out period, the LSTM beat both
baselines on only 2 of 5 tickers (MSFT, NVDA), with directional accuracy
(did it correctly call up vs. down) in the 40-60% range — close to a coin
flip. I report this honestly rather than only showing favorable results.

**Q: Isn't a 40-60% directional accuracy basically useless?**

It's a fair critique, and I'd frame it this way: it means the model's output
should be read as a probabilistic lean, not a confident prediction — which
is exactly why I pair it with the GenAI narrative layer, which adds
qualitative context a pure price-history model can't see. It's also
consistent with efficient-markets intuition — if 7-day stock moves were
easily predictable from price history alone, that edge would already be
arbitraged away by every quant fund doing the same thing. A model that
looked *suspiciously* good at this would be a bigger red flag to me than
these honest, modest results.

**Q: What's the difference between your naive and drift baselines?**

Naive: assume tomorrow's price equals today's (a flat line). Drift: assume
the recent average daily return continues (extrapolate the trend). Drift is
a slightly smarter baseline; in my results it actually wasn't always better
than naive, which itself is informative about how choppy vs. trending each
stock's recent behavior was.

**Q: Tell me about a time you found a bug or issue in your own model's
behavior and how you diagnosed it.**

I noticed one specific 30-day backtest window for AMZN had drastically worse
price-magnitude error (~58% worse than baseline) while directional accuracy
in that same window was actually strong. That "right direction, wrong
magnitude" pattern pointed at a calibration issue rather than a general
model failure. I cross-referenced the date range against AMZN's actual
earnings calendar and confirmed it aligned exactly with their Q1 2026
earnings report — a scheduled volatility event the model, which only sees
price history, had no way to anticipate. That diagnosis directly motivated
the earnings-feature experiment described above.

**Q: You tested a fix and it didn't work. Why include that in your writeup /
resume instead of leaving it out?**

Because the negative result is genuine evidence of process, not just
outcome. Anyone can report a metric that went up. Showing a hypothesis,
a controlled test, and an honest "this didn't work, here's my best
explanation why" demonstrates I actually validate my own work rather than
just shipping whatever seems to run. I'd rather an interviewer see that
than a suspiciously clean success story.

**Q: How did you make sure your ablation test was actually controlled?**

I fixed the random seed (`torch.manual_seed`, `np.random.seed`) immediately
before model initialization in both runs, so the only variable that changed
between the 6-feature and 8-feature versions was the feature set itself —
not random weight initialization luck. Early in the project I ran a
comparison *without* controlling for this and got a misleading result (the
earnings feature looked like it helped) — catching that and rerunning
properly was itself a useful lesson about what "controlled experiment"
actually requires in ML.

**Q: You also tested different amounts of training data. What did you find?**

I compared 2, 5, and 10 years of training history across all 5 tickers.
There was no universal winner — AAPL and MSFT actually moved in *opposite*
directions as I added more data (AAPL improved, MSFT got slightly worse),
which rules out a simple "more data is always better" story. My best
explanation is that the right amount of history is ticker-specific, likely
tied to how much a stock's longer-term price behavior includes regime shifts
(e.g., AMZN's pandemic-era hypergrowth-and-correction cycle) that dilute
more relevant recent patterns. I chose 10 years as the shipped default since
it was best-or-near-best on 3 of 5 tickers, but I documented this as a
finding rather than pretending there was one clean answer.

**Q: Is there a ticker your model never gets right?**

Yes — GOOGL. It underperforms both baselines at every data volume I tested
(2yr, 5yr, 10yr), including losing in 4 of 4 rolling backtest windows at one
point. Since changing the data volume didn't move this result at all, I
concluded the issue isn't a data-volume problem — it's something about the
feature set, hyperparameters, or GOOGL's specific price dynamics that this
architecture doesn't capture well. I flagged it as an open, unresolved
limitation rather than continuing to chase it with diminishing returns.

---

## 4. RAG / GenAI questions

**Q: Explain how the RAG component works.**

News articles get fetched via Tavily and converted into vector embeddings
(numerical representations of meaning) using a sentence-transformer model,
stored in a FAISS index. When a user requests analysis for a ticker, a query
gets embedded the same way, and FAISS finds the most semantically similar
articles — better than keyword search since it can match on meaning, not
exact words. Those retrieved articles get passed into the LLM prompt
alongside the LSTM's forecast, and Mistral generates a structured JSON
verdict: sentiment, risks, catalysts, recommendation, and a rationale tying
the two together.

**Q: Is RAG actually necessary here, or overkill?**

Honestly, at my current scale (a few dozen articles across 5 tickers),
plain filtering by ticker would probably work almost as well — the corpus
is small enough that vector search isn't solving a real scale problem yet.
I'd say it demonstrates I can build the pattern correctly, more than it's
strictly necessary at this size. It would become genuinely necessary if this
scaled to hundreds of tickers and years of news, where you couldn't fit
everything into a prompt and actually need retrieval to narrow the field.
I'd rather say that plainly than overclaim its necessity.

**Q: Does the news affect the price forecast?**

No, and this is an important distinction in my design — the LSTM's price
forecast is generated purely from historical price/volume/technical-indicator
data, with zero code path to the news system. News only feeds into the
*narrative* layer, generated by the LLM *after* the price forecast already
exists. This is actually connected to the earnings-magnitude problem I
found — since the forecasting model never sees news, it has no way to know
a volatility-inducing event like earnings is coming.

**Q: How do you handle the LLM returning malformed output?**

The prompt explicitly instructs the model to return only a JSON object with
a fixed schema, and I wrap the parsing in a try/except — if
`json.loads()` fails, I return a fallback response (sentiment: "unknown",
recommendation: "hold") with the raw text included for debugging, rather
than letting the whole request crash.

**Q: I noticed news cards sometimes showed articles about a different
company than requested — what happened and how did you fix it?**

Two separate issues, actually. First, duplicate articles: broad
market-recap pieces were legitimately retrieved for multiple tickers'
searches, so I added deduplication by URL across the whole corpus at
index-build time. Second, genuine cross-ticker bleed: my retriever searched
the entire FAISS index and just re-sorted results to prefer exact matches,
but if there weren't enough truly-relevant articles for a ticker within a
small top-k search, other companies' articles filled the gap. I fixed this
by widening the initial candidate pool substantially, then filtering
strictly to the requested ticker (plus macro-tagged articles), only falling
back to other tickers as an explicit last resort — meaning you now see
fewer, more correct results rather than a padded, less-relevant list.

---

## 5. System design / engineering questions

**Q: Walk me through your system architecture.**

FastAPI backend with three main capabilities behind separate endpoints:
`/forecast` (LSTM inference only), `/news-analysis` (RAG retrieval only),
and `/report` (both plus LLM synthesis). A vanilla JS/HTML dashboard calls
these directly — no framework needed for this scope. Deployed as a single
Docker container on Hugging Face Spaces, which serves both the API and the
static frontend from one FastAPI app, so there's one deployed URL rather
than separately hosting frontend and backend.

**Q: What production issues did you actually run into deploying this?**

A few real ones: Yahoo Finance started rate-limiting requests from the
shared cloud IP once deployed (works fine locally where you're not sharing
an IP with hundreds of other apps) — I fixed this with an in-memory cache
with a 1-hour TTL, since a 7-day forecast doesn't need up-to-the-second
data, plus a graceful fallback to stale cached data instead of crashing.
Separately, I found that my LSTM used Monte Carlo dropout, meaning every
call produced slightly different random results — so the chart and the
LLM's narrative, which independently called the forecast function, would
show inconsistent numbers. I fixed that with a short-lived cache so repeat
calls within a window return the identical result.

**Q: How do you handle secrets/API keys?**

Environment variables loaded via `python-dotenv` locally (`.env`, gitignored),
and injected as platform secrets in production (Hugging Face Spaces'
repository secrets). Same `os.environ.get()` call works in both
environments without any code branching.

**Q: Why deploy on Hugging Face Spaces instead of something like AWS?**

Practical reasons: it's free without requiring a credit card (some
alternatives like Render started requiring card verification), and it's
genuinely well-suited to ML workloads — Docker-based, comfortable with
heavier dependencies like PyTorch and FAISS. For a portfolio project, the
deployment platform matters less than demonstrating I can actually get a
real multi-service ML system into a publicly accessible, working state.

---

## 6. Honest self-critique questions (be ready for these — don't get defensive)

**Q: What would you do differently if you started over?**

Two things: I'd add a fixed random seed from the very start rather than
discovering the need for it mid-project after an uncontrolled comparison
gave me a misleading result. And I'd build the backtesting harness *before*
adding features like the earnings signal, so I'm testing hypotheses against
a rigorous baseline from day one instead of retrofitting rigor afterward.

**Q: This sounds like a lot of negative results. Did the project actually
succeed?**

I'd push back gently on "negative" — the project succeeded at what it was
actually for, which was demonstrating I can build a full ML system and
evaluate it honestly. If the goal had been "produce a stock-picking tool
that works," then no, it doesn't clear that bar, and I say so directly. But
that was never a credible goal for a project built in this timeframe on
this much data — professional quant teams with vastly more resources don't
have a guaranteed edge here either.

**Q: Would you actually trade real money based on this model?**

No, and I'm explicit about that in my documentation. Directional accuracy
near a coin flip and inconsistent baseline performance across tickers mean
this isn't close to production-grade for real capital decisions, which
would need transaction-cost modeling, walk-forward retraining, proper risk
management, and far more historical validation than a portfolio project
scope allows.

**Q: What's the single biggest limitation of this project?**

Data volume, most likely — several of my findings (the earnings feature
failing, GOOGL underperforming regardless of settings) are consistent with
simply not having enough training examples per ticker for the model to
learn reliable patterns, especially for anything with a low base rate. A
production version of this would likely need either far more historical
data, pooled training across many tickers instead of one model per ticker,
or both.

---

## Quick-reference: numbers to have ready

- 5 tickers (AAPL, MSFT, GOOGL, NVDA, AMZN), 10 years daily data, ~1,970
  training sequences per ticker
- LSTM beats baselines on 2/5 tickers; directional accuracy 40-60%
- Earnings-feature ablation: 6-feature model beat 8-feature (with earnings
  signal) model by a wide margin in the target window (23% vs 59% worse
  than baseline) — feature rejected
- Data-volume sweep: 2yr/5yr/10yr tested; no universal winner; 10yr chosen
  as default
- GOOGL underperforms at every data volume tested — open limitation

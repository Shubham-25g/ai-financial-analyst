# The Complete Guide to This Project

This document explains every piece of this project — the concepts behind it,
how data flows through it, why each decision was made, and what every file
does. Read it top to bottom once, then use it as a reference.

---

## Part 1: The core idea, in plain terms

This project answers a question in two independent ways and then combines
the answers:

1. **"Where might this stock's price go in the next week?"** — answered by
   a neural network trained on historical price data (no news, no opinions,
   just numbers).
2. **"What's actually happening with this company right now, and does it
   support or contradict that number?"** — answered by an AI language model
   reading real news articles.

These two answers are generated **completely independently** and only meet
at the very last step, when an LLM is asked to look at both and write a
verdict. This separation is deliberate and important — it means the price
forecast is never influenced by news (a design choice with real
consequences, explained in Part 4).

---

## Part 2: Concepts you need before anything else makes sense

### What is an LSTM, actually?
LSTM stands for Long Short-Term Memory. It's a type of neural network built
for **sequences** — data where order matters (like a sentence, or 30 days
of stock prices). Unlike a normal neural network that looks at all its
inputs at once with no sense of order, an LSTM reads through a sequence one
step at a time, carrying forward an internal "memory" that gets updated at
each step. This lets it learn things like "this is the third day of a
downtrend" rather than just seeing 30 unordered numbers.

**Why it matters here:** stock prices are inherently sequential — today's
price is connected to yesterday's, and the last 30 days as a *pattern*
(not just as 30 separate facts) is what the model is trying to learn from.

### What is a "residual," and why predict one instead of the real price?
A residual is "the part that's left over after you subtract out what's
easy to explain." In this project: instead of asking the LSTM to predict
the stock's actual future price, we first compute a simple **drift
baseline** — "just assume the recent average daily trend continues" — and
then train the LSTM to predict only the *difference* between what actually
happens and what that simple baseline would have guessed.

**Why:** the drift baseline already captures most of what's predictable
about a trending stock, for free, with no machine learning at all. If we
trained the LSTM on raw prices, it would spend most of its learning
capacity rediscovering "prices tend to keep trending" — something two lines
of arithmetic already know. Making it predict the residual forces it to
find the *harder*, more valuable signal.

### What is Monte Carlo Dropout, and why does it give a confidence range?
"Dropout" is a training technique where a neural network randomly
"turns off" some of its internal connections during training, which helps
prevent it from over-relying on any single pathway (this is normally
disabled once training is done). Monte Carlo Dropout is a trick where you
**deliberately leave dropout turned on at prediction time too**, and run
the same prediction multiple times (here, 30 times). Because different
random connections get dropped each time, you get 30 *slightly different*
predictions instead of one.

**Why it matters:** the spread of those 30 predictions tells you how
uncertain the model is. If all 30 runs agree closely, the model is
confident. If they're spread wide apart, the model itself doesn't have a
clear signal. This project uses the middle (median) of the 30 as the
headline forecast, and the 10th/90th percentile as the confidence band you
see shaded on the chart.

### What is RAG (Retrieval-Augmented Generation)?
A language model like Mistral only knows what it learned during training —
it has no idea what happened in the news this week. RAG solves this by:
**Retrieving** relevant documents from a search system, then **Augmenting**
the AI's prompt with that retrieved text, so it **Generates** its answer
grounded in real, current information instead of guessing from stale
training knowledge.

### What is a vector embedding, and what is FAISS?
An embedding is a way of turning text into a list of numbers (a vector)
that represents its *meaning* — texts with similar meaning end up with
similar-looking vectors, even if they don't share the same words. FAISS is
a library for very efficiently searching through thousands/millions of
these vectors to find the ones closest in meaning to a query vector — this
is what makes "search by meaning" (semantic search) possible instead of
just matching exact keywords.

---

## Part 3: The complete data flow, stage by stage

### Stage 0 — Nothing runs without these three external services
- **Yahoo Finance** (via the free `yfinance` Python library) — real stock
  price history, no account needed.
- **Tavily** — a search API specifically built for feeding AI systems
  current web/news information.
- **Mistral** — the AI language model that writes the final narrative.

### Stage 1 — Getting raw price data
`backend/app/data/market_data.py::fetch_ohlcv()` calls `yfinance` and
downloads 10 years of daily Open/High/Low/Close/Volume data for one ticker.
This function is cached for an hour (added after a real production issue —
see Part 6) so repeated requests don't hammer Yahoo Finance's servers.

### Stage 2 — Turning raw prices into model-ready features
`market_data.py::add_technical_indicators()` computes 5 additional columns
from the raw price series:
- **SMA (Simple Moving Average)**, 10-day and 30-day — smoothed price
  trends that filter out daily noise
- **RSI (Relative Strength Index)** — a momentum indicator showing whether
  a stock has been bought or sold aggressively recently (0-100 scale)
- **MACD** — the difference between two different moving averages,
  commonly used to gauge trend strength/direction

Combined with `close` and `volume`, this gives the model 6 input features
per day.

*(Note: `earnings_calendar.py` also exists and can compute earnings-proximity
features — this was built, tested, and deliberately excluded from the final
feature set after it was shown to hurt performance. Full story in
RESULTS.md.)*

### Stage 3 — Turning a price history into trainable examples
`backend/app/models/train.py::build_sequences()` is where raw data becomes
something a neural network can actually learn from. For every day `i` in
the historical data, it creates one training example:
- **Input**: the preceding 30 days of the 6 features
- **Target**: the residual-over-drift for the next 7 days (see Part 2)

With ~10 years of data, this produces roughly 1,970 training examples per
ticker.

### Stage 4 — The model architecture
`backend/app/models/lstm_model.py` defines the network: an LSTM layer reads
the 30-day input sequence and produces a final "memory" summary, which then
passes through a small feedforward head that outputs 7 numbers — the
predicted residual return for each of the next 7 days.

### Stage 5 — Training
`train.py::train_one_ticker()` runs the standard supervised learning loop:
1. Split data 85% training / 15% validation
2. For 60 passes (epochs): show the model training examples, measure how
   wrong it was (Mean Squared Error), and adjust its internal weights
   slightly to do better (via the Adam optimizer)
3. After every epoch, check performance on the *validation* data (which the
   model never trains on) — this catches overfitting, where a model gets
   great at memorizing training data but worse at generalizing
4. Save only the version of the model that scored best on validation data,
   not necessarily the final epoch

Output: one `.pt` checkpoint file per ticker, containing the trained
weights plus metadata needed to use the model later (normalization
statistics, feature list, etc.).

### Stage 6 — Turning a trained model into a forecast
`backend/app/models/predict.py::forecast()`:
1. Loads a ticker's checkpoint
2. Feeds it the most recent 30 days of data
3. Runs it 30 times with Monte Carlo Dropout active (see Part 2)
4. Adds the drift baseline back to each of the 30 residual predictions to
   get actual predicted prices
5. Takes the median as the headline forecast, and the 10th/90th percentile
   as the confidence band
6. Caches the result for 10 minutes, so repeated requests (e.g. the chart
   display and the LLM narrative, which both need a forecast) get the
   *exact same* numbers instead of two different random MC Dropout samples

### Stage 7 — Retrieving relevant news
`backend/app/rag/news_corpus.py::fetch_live_news()` searches Tavily for
recent news about a ticker, cleans up the scraped text (strips markdown
noise, caps length), and captures the article's URL. `all_documents()`
pulls this for every tracked ticker plus a general macro-economic query,
and **deduplicates** across all of them by URL — otherwise a broad
market-recap article would show up multiple times under different tickers.

### Stage 8 — Building the searchable index
`vector_store.py::build_index()` converts every article into an embedding
(via `sentence-transformers`) and stores them in a FAISS index, saved to
disk. This only needs to be rebuilt when the news corpus changes.

### Stage 9 — Retrieving news for a specific ticker
`retriever.py::retrieve_for_ticker()` embeds a query like "NVDA stock
recent earnings news sentiment risk," searches a *wide* pool of candidates
in FAISS, then filters strictly: real matches for this ticker first, then
macro-context articles, and only falls back to other companies' articles as
an explicit last resort if there genuinely isn't enough relevant material.

### Stage 10 — Synthesis: combining forecast + news into a verdict
`synthesizer.py::synthesize()` builds a prompt containing the LSTM's 7-day
forecast numbers and the retrieved news text, sends it to Mistral with
instructions to return structured JSON (sentiment, risks, catalysts,
recommendation, confidence, and a plain-English rationale), and parses the
response — falling back gracefully if the model doesn't return valid JSON.

### Stage 11 — Serving it all over HTTP
`backend/app/main.py` and `routers/` expose three main endpoints:
- `/forecast/{ticker}` — Stage 6 only (fast, no LLM call)
- `/news-analysis/{ticker}` — Stages 7-9 only (fast, no LLM call)
- `/report/{ticker}` — everything combined (slower, includes the LLM call)

The same FastAPI app also serves the dashboard's static files, so one
deployed service handles both the API and the UI.

### Stage 12 — The dashboard
`frontend/index.html` is a single-page app (vanilla HTML/CSS/JS, no
framework) that calls the endpoints above and renders: a price chart with
confidence bands (Chart.js), news cards, and a verdict panel. It requests
`/forecast` and `/news-analysis` in parallel with `/report`, so the chart
and news render immediately rather than waiting on the slow LLM call to
finish (a real fix made after noticing this exact lag — see Part 6).

---

## Part 4: Key design decisions and why they were made

**Why is the LSTM never shown any news?**
This was a deliberate architectural choice, not an oversight — it keeps the
two signals cleanly separable (you can always tell whether a forecast came
from "just the numbers" or "the numbers plus context"). The trade-off,
discovered through backtesting, is that the model has zero way to anticipate
scheduled events like earnings reports, which directly caused a diagnosed
failure mode (see RESULTS.md — the AMZN earnings-window magnitude miss).

**Why predict a residual instead of raw price/returns?**
Covered in Part 2 — it focuses the model's learning capacity on the harder,
more valuable part of the problem instead of re-deriving what a simple
baseline already knows.

**Why Monte Carlo Dropout instead of a "real" ensemble of models?**
Training multiple separate models (a true ensemble) is more principled but
far more expensive — you'd need to train and store N complete models
instead of one. MC Dropout approximates the *benefit* of an ensemble
(multiple slightly different opinions) using only one trained model,
by exploiting randomness that's already built into the architecture.

**Why FAISS instead of a hosted vector database?**
FAISS runs in-process with no separate server to manage — appropriate for
this project's scale (a few dozen articles). At real production scale
(thousands of tickers, years of news), a hosted solution like Qdrant would
make more sense; `vector_store.py` is deliberately a thin wrapper so that
swap is contained to one file.

**Why Mistral instead of building a local/offline LLM?**
API-based LLMs are dramatically simpler to integrate and produce
higher-quality structured output than a small local model would, for a
fraction of the engineering effort. The trade-off is an external dependency
and API cost — acceptable for this project's scope.

---

## Part 5: What the numbers actually mean (see RESULTS.md for full detail)

- The LSTM beats simple baselines (naive "no change," drift "extrapolate
  trend") on only **2 of 5 tickers** tested, with directional accuracy
  (correctly calling up vs. down) around **40-60%** — close to a coin flip.
- An earnings-proximity feature was built, hypothesized to help, tested
  with a controlled experiment (same random seed, only the feature set
  changed), and found to **hurt** performance — rejected rather than shipped.
- A sweep of training-data volume (2/5/10 years) found no universal answer
  — some tickers improved with more history, one got worse — meaning the
  "right" amount of training data is ticker-specific, not a fixed rule.
- One ticker (GOOGL) underperforms baselines regardless of any setting
  tested — an open, documented limitation, not something more data or
  features currently fix.

**The honest bottom line:** this system is not reliable enough for real
trading decisions. Its value is as a demonstration of building a complete
ML + GenAI pipeline *and* rigorously, honestly evaluating it — which is a
rarer and more valuable skill to show than a model that happens to work.

---

## Part 6: Real production issues that came up, and how they were fixed

**Yahoo Finance rate limiting.** Once deployed, the app shares an outbound
IP address with many other cloud-hosted apps, and Yahoo Finance started
blocking requests as suspicious traffic. Fixed with an hour-long cache per
ticker (a 7-day forecast doesn't need up-to-the-second freshness) plus a
graceful fallback to stale cached data instead of crashing.

**Inconsistent numbers between the chart and the AI narrative.** Because
the forecast function uses random Monte Carlo Dropout sampling, two
separate calls to it (one for the chart, one triggered internally by the
report endpoint) produced two different random results. Fixed with a
10-minute cache so repeat calls return the identical forecast.

**The chart appeared to hang when generating a full report.** The frontend
was waiting for the *entire* report (forecast + news + the slow LLM call)
to finish before displaying anything, even though the forecast itself was
ready almost instantly. Fixed by firing the forecast/news requests and the
full-report request in parallel, rendering the chart and news the moment
they're ready.

**Cross-ticker news bleed.** The news retriever searched the whole index
and just re-sorted results, so if there weren't enough genuinely relevant
articles for a ticker, other companies' articles filled the remaining
slots. Fixed by searching a much wider candidate pool first, then filtering
strictly to the requested ticker (plus macro content), only padding with
other tickers as an explicit last resort.

---

## Part 7: How to actually run and explore this yourself

1. `python -m app.models.train` — trains all 5 tickers, takes a few minutes
2. `python -m app.rag.vector_store --build` — builds the news search index
3. `uvicorn app.main:app --reload --port 8000` — runs everything
4. Open `http://localhost:8000` — the dashboard
5. `python scripts/backtest.py --all --test-days 120 --window 30` — see the
   rigor layer in action yourself
6. `/docs` on the running server — FastAPI's interactive API explorer, lets
   you call any endpoint directly and see raw JSON responses

---

## Glossary

| Term | Meaning |
|---|---|
| LSTM | A neural network architecture designed to learn from sequences/ordered data |
| Epoch | One complete pass through the training data during training |
| Overfitting | When a model memorizes training data instead of learning generalizable patterns |
| Residual | The part of a value not explained by a simpler baseline |
| Drift baseline | Forecasting method: extrapolate the recent average trend |
| Monte Carlo Dropout | Running a model multiple times with randomness left on, to estimate uncertainty |
| RAG | Retrieval-Augmented Generation — grounding an LLM's answer in retrieved real documents |
| Embedding | A numerical vector representing the meaning of a piece of text |
| FAISS | A library for fast similarity search over large sets of embeddings |
| Backtest | Testing a forecasting model against historical data it wasn't trained on, to see how it would have performed |
| Ablation | An experiment that removes/changes one variable to isolate its effect |
| Directional accuracy | The percentage of times a forecast correctly predicted the direction (up/down) of a move, regardless of magnitude |

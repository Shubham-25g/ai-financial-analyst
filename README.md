---
title: AI Financial Analyst
emoji: 📈
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Ledger.ai — AI Financial Analyst

A hybrid forecasting dashboard that pairs a **deep learning time-series
model (LSTM)** for quantitative price forecasting with a **RAG + LLM
pipeline** that reads real news and synthesizes a buy/sell/hold narrative —
combining the two into one view.

> **Read this first:** [RESULTS.md](RESULTS.md) documents how this was
> rigorously backtested against baseline models, and an honest account of
> a feature hypothesis that was tested and rejected. This project's main
> value isn't "it predicts stocks" — it's the evidence-based process behind
> figuring out where it does and doesn't actually work.

## What it does

- **Forecasts** a stock's price 7 days out using an LSTM trained on real
  historical price/volume/technical-indicator data (`yfinance`)
- **Reads recent news** for that ticker via retrieval-augmented generation
  (Tavily for search, FAISS for retrieval)
- **Synthesizes both** into a buy/sell/hold call with a plain-English
  rationale, using Mistral
- Presents all of it in a live dashboard — forecast chart with confidence
  bands, retrieved news cards with source links, and a GenAI verdict panel

## Architecture

```
┌──────────────────┐        ┌──────────────────────┐
│  Market Data      │──────▶│  LSTM Forecaster      │──┐
│  (yfinance + TA)  │       │  (PyTorch)            │  │
└──────────────────┘        └──────────────────────┘  │      ┌───────────────────┐
                                                         ├────▶│  Synthesis Layer   │──▶ Dashboard
┌──────────────────┐        ┌──────────────────────┐  │      │  (Mistral LLM)     │
│  News (Tavily)    │──────▶│  RAG Pipeline         │──┘      └───────────────────┘
│                   │       │  (FAISS + embeddings) │
└──────────────────┘        └──────────────────────┘
```

**Important distinction:** news and price forecasting are independent —
the LSTM never sees news, only price history. News only shapes the
narrative layer generated *after* the price forecast already exists. See
`RESULTS.md` for why this matters (it's directly tied to a diagnosed model
limitation around earnings events).

## Tech stack

| Layer | Tool |
|---|---|
| Time-series forecasting | PyTorch (LSTM) |
| Market data | `yfinance` |
| News retrieval | Tavily |
| Embeddings + vector search | `sentence-transformers` + FAISS |
| LLM synthesis | Mistral (via `langchain-mistralai`) |
| API | FastAPI |
| Dashboard | Vanilla HTML/JS + Chart.js |

## Key findings (see [RESULTS.md](RESULTS.md) for full detail)

- Backtested against naive and drift baselines across 5 tickers and
  multiple rolling time windows — the LSTM beats baselines on 2/5 tickers,
  with directional accuracy in the 40-60% range overall.
- Diagnosed a magnitude-miscalibration pattern around AMZN's earnings
  report date; built and tested an earnings-proximity feature to address
  it; **rejected it** after a controlled ablation showed it hurt
  performance rather than helping.
- Ran a 2yr/5yr/10yr training-data sweep; found the ideal amount of
  training history is ticker-dependent, not universal — 10 years was
  chosen as the shipped default as the best overall compromise.

## Project structure

```
backend/app/
  config.py                # tickers, hyperparameters, feature list
  data/
    market_data.py          # yfinance fetch + technical indicators
    earnings_calendar.py    # earnings-proximity feature (tested, not used — see RESULTS.md)
  models/
    lstm_model.py           # LSTM architecture
    baselines.py            # naive + drift baseline math (shared by train/predict/backtest)
    train.py                # training loop, saves checkpoints per ticker
    predict.py              # 7-day inference with Monte Carlo dropout confidence bands
  rag/
    news_corpus.py          # Tavily fetch + dedup + cleanup
    vector_store.py         # FAISS index build/query
    retriever.py            # ticker-aware retrieval
    synthesizer.py          # LLM call -> buy/sell/hold JSON
  routers/                  # FastAPI endpoints
  main.py                   # FastAPI app (also serves the frontend)
frontend/
  index.html                # dashboard (chart, news cards, verdict panel)
scripts/
  run_pipeline.py            # one-shot CLI demo: train -> forecast -> report
  backtest.py                 # walk-forward backtest vs. baselines, with rolling-window mode
RESULTS.md                  # full methodology + honest findings
```

## Setup (local)

```bash
# 1. environment
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt

# 2. secrets — create a .env file in the project root
echo "MISTRAL_API_KEY=your_key_here" >> .env
echo "TAVILY_API_KEY=your_key_here" >> .env
echo "HF_TOKEN=your_token_here" >> .env      # optional, silences a rate-limit warning

# 3. train models (per ticker, a few minutes on CPU)
cd backend
python -m app.models.train

# 4. build the news/RAG index
python -m app.rag.vector_store --build

# 5. run
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** — the dashboard and API are served
from the same address.

Or run the whole pipeline in one shot:
```bash
python scripts/run_pipeline.py --ticker NVDA
```

## Backtesting

```bash
python scripts/backtest.py --all --test-days 120 --window 30
```
Compares the LSTM against naive and drift baselines, with a rolling
sub-window view to check whether results are consistent over time or a
one-off stretch. See `RESULTS.md` for how to read the output.

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /tickers` | tracked ticker list |
| `GET /forecast/{ticker}` | 7-day LSTM forecast + confidence band |
| `GET /news-analysis/{ticker}` | RAG-retrieved news for this ticker |
| `GET /report/{ticker}` | forecast + news + LLM-synthesized buy/sell/hold call |
| `GET /docs` | interactive API explorer (FastAPI/Swagger) |

## Data source notes

- **Market data**: real, via `yfinance` — 10 years daily OHLCV per ticker.
- **News**: real, via Tavily — deduplicated across tickers, lightly
  cleaned of scraped markdown/nav noise, with source links.
- **Earnings calendar**: real, via `yfinance`'s earnings-date lookup, with
  an offline synthetic-quarterly fallback if that call fails (it can be
  inconsistent across tickers — see `RESULTS.md`).

## Known limitations

- Directional accuracy hovers near a coin flip (40-60%) — read the model's
  output as a probabilistic lean, not a confident prediction.
- GOOGL consistently underperforms baselines regardless of training data
  volume — a documented, unresolved limitation, not something more data
  fixes (see `RESULTS.md`).
- This is a portfolio/learning project, not a trading tool. Real capital
  decisions need far more rigor (transaction costs, walk-forward
  retraining, risk management) than is in scope here.

## Deployment

Deployed via Hugging Face Spaces (Docker runtime) — see the `Dockerfile`
in this repo. Requires the three env vars above set as Space secrets.
Alternative: any host that can run a long-lived Python/Docker process
(Render, Railway, Fly.io) will work with minor config changes.

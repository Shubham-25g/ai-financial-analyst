"""
One-shot demo: trains a model for the given ticker if no checkpoint exists,
builds the RAG index if needed, runs a forecast, retrieves news, and (if
ANTHROPIC_API_KEY is set) synthesizes a buy/sell/hold report.

Usage: python scripts/run_pipeline.py --ticker NVDA
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app import config
from app.models.train import train_one_ticker
from app.models.predict import forecast as run_forecast
from app.rag.vector_store import build_index
from app.rag.retriever import retrieve_for_ticker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    ckpt_path = config.MODELS_DIR / f"{ticker}.pt"
    if args.force_retrain or not ckpt_path.exists():
        print(f"--- Training LSTM for {ticker} ---")
        train_one_ticker(ticker)

    if not config.VECTOR_INDEX_PATH.exists():
        print("--- Building RAG index ---")
        build_index()

    print(f"\n--- 7-day forecast for {ticker} ---")
    fc = run_forecast(ticker)
    print(json.dumps(fc, indent=2))

    print(f"\n--- Retrieved news for {ticker} ---")
    news = retrieve_for_ticker(ticker, k=5)
    for n in news:
        print(f"  [{n['relevance_score']:.3f}] {n['headline']}")

    if config.ANTHROPIC_API_KEY:
        from app.rag.synthesizer import synthesize
        print(f"\n--- GenAI synthesis for {ticker} ---")
        analysis = synthesize(ticker, fc, news)
        print(json.dumps(analysis, indent=2))
    else:
        print("\n(Set ANTHROPIC_API_KEY to also run the GenAI synthesis step.)")


if __name__ == "__main__":
    main()

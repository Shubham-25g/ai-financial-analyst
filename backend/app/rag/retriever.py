"""Ticker-aware retrieval: builds a query from the ticker + generic financial
risk themes, retrieves top-k relevant news chunks via the vector store."""
from __future__ import annotations
from app.rag import vector_store


def retrieve_for_ticker(ticker: str, k: int = 5) -> list[dict]:
    query = (
        f"{ticker} stock recent earnings news sentiment risk macroeconomic "
        f"factors affecting price"
    )
    results = vector_store.search(query, k=k)
    # Prefer ticker-specific docs first, then macro context
    results.sort(key=lambda d: (d["ticker"] != ticker, -d["relevance_score"]))
    return results

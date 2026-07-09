"""Ticker-aware retrieval: builds a query from the ticker + generic financial
risk themes, retrieves top-k relevant news chunks via the vector store."""
from __future__ import annotations
from app.rag import vector_store


def retrieve_for_ticker(ticker: str, k: int = 5) -> list[dict]:
    query = (
        f"{ticker} stock recent earnings news sentiment risk macroeconomic "
        f"factors affecting price"
    )

    # Search a wider candidate pool than we actually need. The index holds
    # articles for every tracked ticker, and a small top-k similarity search
    # can end up dominated by semantically-similar articles about OTHER
    # companies (e.g. a broad "AI stocks rally" piece scoring high for every
    # tech ticker's query) rather than ones genuinely about this ticker.
    # Pulling a larger pool and filtering strictly fixes that.
    candidate_pool_size = max(k * 4, 20)
    results = vector_store.search(query, k=candidate_pool_size)

    exact_matches = [d for d in results if d["ticker"] == ticker]
    macro_matches = [d for d in results if d["ticker"] == "MACRO"]
    other_matches = [d for d in results if d["ticker"] not in (ticker, "MACRO")]

    # Prefer ticker-specific articles, then macro context. Only fall back to
    # other-company articles if there genuinely isn't enough relevant material
    # for this ticker — better to show fewer, correct results than pad the
    # list with unrelated companies.
    combined = exact_matches + macro_matches
    if len(combined) < k:
        combined += other_matches[: k - len(combined)]

    return combined[:k]


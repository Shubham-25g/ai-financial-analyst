"""
News/earnings-transcript corpus for RAG.

`fetch_live_news()` pulls from Tavily. `all_documents()` is called once at
index-build time to gather everything that goes into the FAISS index —
it also deduplicates across tickers, since broad market-recap articles
(e.g. "Stock Market Today: ...") legitimately surface for multiple
per-ticker searches and would otherwise show up as near-duplicate cards
in the UI.
"""
from __future__ import annotations
import os
import re
from datetime import datetime, timedelta

from app import config

SAMPLE_NEWS = [
    {
        "ticker": "NVDA", "date": "2026-06-28", "url": None,
        "headline": "NVIDIA data-center demand remains strong amid AI capex cycle",
        "text": (
            "NVIDIA's data-center segment continues to see robust demand as "
            "hyperscalers extend multi-year AI infrastructure buildouts. Analysts "
            "note supply constraints on advanced packaging remain the key "
            "bottleneck rather than demand softness."
        ),
    },
    {
        "ticker": "AAPL", "date": "2026-06-25", "url": None,
        "headline": "Apple services growth offsets soft hardware upgrade cycle",
        "text": (
            "Apple's services revenue continues to grow at a healthy clip, "
            "partially offsetting a more muted iPhone upgrade cycle in several "
            "key markets."
        ),
    },
    {
        "ticker": "MACRO", "date": "2026-07-01", "url": None,
        "headline": "Fed signals patience on rate cuts amid mixed inflation data",
        "text": (
            "The Federal Reserve signaled a patient, data-dependent approach to "
            "further rate cuts after mixed inflation readings."
        ),
    },
]


def _clean_text(text: str, max_chars: int = 600) -> str:
    """
    Light cleanup of Tavily-scraped article text: strips markdown headers/
    hashes, collapses excess whitespace, and caps length. This doesn't fix
    genuinely low-quality scrapes (e.g. nav-link-heavy pages), but it removes
    the most common noise and keeps cards a predictable, readable length.
    """
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)   # markdown headers
    text = re.sub(r"\s+", " ", text).strip()                  # collapse whitespace
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def fetch_live_news(ticker: str, days_back: int = 14, tag: str | None = None):
    """
    Fetch recent news via Tavily.

    `ticker` is used as the search query (can be a real ticker like "NVDA"
    or a free-text query like "Federal Reserve interest rates").
    `tag` overrides what gets stored in the "ticker" field of each returned
    doc — use this for non-ticker queries (e.g. macro news).
    """
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set TAVILY_API_KEY in your environment/.env file.")

    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=f"{ticker} stock news earnings recent developments",
        topic="news",
        days=days_back,
        max_results=8,
        include_answer=False,
    )

    label = tag or ticker
    today_str = datetime.today().strftime("%Y-%m-%d")

    return [
        {
            "ticker": label,
            "date": (r.get("published_date") or "")[:10] or today_str,
            "headline": r["title"],
            "text": _clean_text(r.get("content", "")),
            "url": r.get("url"),
        }
        for r in resp.get("results", [])
    ]


def all_documents():
    """
    Pulls live news for every tracked ticker + a macro-only query, for
    indexing. Deduplicates by URL (falling back to headline if a result
    has no URL) so a broad market-recap article that surfaces under
    several tickers' searches only appears once in the index.
    """
    docs = []
    seen_keys = set()

    def _add(doc):
        key = doc.get("url") or doc["headline"]
        if key in seen_keys:
            return
        seen_keys.add(key)
        docs.append(doc)

    for ticker in config.TICKERS:
        try:
            for doc in fetch_live_news(ticker):
                _add(doc)
        except Exception as e:
            print(f"[news_corpus] Failed to fetch news for {ticker}: {e}")

    try:
        for doc in fetch_live_news(
            "macroeconomic outlook Federal Reserve interest rates", tag="MACRO"
        ):
            _add(doc)
    except Exception as e:
        print(f"[news_corpus] Failed to fetch macro news: {e}")

    if not docs:
        print("[news_corpus] No live news fetched — falling back to SAMPLE_NEWS.")
        return SAMPLE_NEWS

    return docs

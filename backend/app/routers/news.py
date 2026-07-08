from fastapi import APIRouter
from app.rag.retriever import retrieve_for_ticker

router = APIRouter()


@router.get("/news-analysis/{ticker}")
def get_news(ticker: str, k: int = 5):
    ticker = ticker.upper()
    return {"ticker": ticker, "news": retrieve_for_ticker(ticker, k=k)}

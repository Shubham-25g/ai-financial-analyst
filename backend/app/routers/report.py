import traceback
from fastapi import APIRouter, HTTPException

from app.models.predict import forecast as run_forecast
from app.rag.retriever import retrieve_for_ticker
from app.rag.synthesizer import synthesize

router = APIRouter()


@router.get("/report/{ticker}")
def get_report(ticker: str, k_news: int = 5):
    ticker = ticker.upper()

    try:
        fc = run_forecast(ticker)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()  # full traceback -> container logs
        raise HTTPException(status_code=500, detail=f"Forecast step failed: {e}")

    try:
        news = retrieve_for_ticker(ticker, k=k_news)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"News retrieval step failed: {e}")

    try:
        analysis = synthesize(ticker, fc, news)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM synthesis step failed: {e}")

    return {
        "ticker": ticker,
        "forecast": fc,
        "news": news,
        "analysis": analysis,
    }
from fastapi import APIRouter, HTTPException
from app.models.predict import forecast as run_forecast
from app.schemas import ForecastResponse

router = APIRouter()


@router.get("/forecast/{ticker}", response_model=ForecastResponse)
def get_forecast(ticker: str):
    ticker = ticker.upper()
    try:
        return run_forecast(ticker)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # Raised by market_data.fetch_ohlcv on yfinance failures/rate limits
        raise HTTPException(status_code=503, detail=str(e))

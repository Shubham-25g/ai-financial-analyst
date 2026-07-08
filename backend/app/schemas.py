from pydantic import BaseModel


class ForecastPoint(BaseModel):
    date: str
    median: float
    low_p10: float
    high_p90: float


class ForecastResponse(BaseModel):
    ticker: str
    last_close: float
    last_date: str
    horizon_days: int
    forecast: list[ForecastPoint]
    expected_return_pct: float


class NewsItem(BaseModel):
    ticker: str
    date: str
    headline: str
    text: str
    relevance_score: float


class SynthesisResponse(BaseModel):
    sentiment: str
    key_risks: list[str]
    key_catalysts: list[str]
    recommendation: str
    confidence: str
    rationale: str


class ReportResponse(BaseModel):
    ticker: str
    forecast: ForecastResponse
    news: list[NewsItem]
    analysis: SynthesisResponse

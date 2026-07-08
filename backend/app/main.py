from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app import config
from app.routers import forecast, news, report

app = FastAPI(
    title="AI Financial Analyst",
    description="LSTM price forecasting + RAG-powered GenAI narrative synthesis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router, tags=["forecast"])
app.include_router(news.router, tags=["news"])
app.include_router(report.router, tags=["report"])


@app.get("/api")
def api_root():
    return {"status": "ok", "message": "AI Financial Analyst API. See /docs for endpoints."}


@app.get("/tickers")
def get_tickers():
    return {"tickers": config.TICKERS}


# Serve the frontend dashboard from the same service, so deployment is a
# single URL rather than two separately-hosted pieces. FRONTEND_DIR resolves
# to ../../frontend relative to this file (backend/app/main.py -> project root/frontend).
FRONTEND_DIR = config.BASE_DIR.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
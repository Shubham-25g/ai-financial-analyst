"""Central configuration: tracked tickers, paths, model hyperparameters."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Sector basket (Tech) — extend/replace freely
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# LSTM hyperparameters
SEQ_LEN = 30            # lookback window (trading days)
FORECAST_HORIZON = 7    # days ahead to predict
DRIFT_LOOKBACK = 60     # days used to estimate the drift baseline the LSTM predicts a residual against
FEATURES = ["close", "volume", "sma_10", "sma_30", "rsi_14", "macd"]#, "days_to_earnings", "earnings_in_horizon"]
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 60
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL = "mistral-medium-latest"

VECTOR_INDEX_PATH = BASE_DIR / "rag" / "faiss.index"
VECTOR_META_PATH = BASE_DIR / "rag" / "faiss_meta.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

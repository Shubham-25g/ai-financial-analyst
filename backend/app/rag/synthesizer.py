"""
Synthesis layer: takes the LSTM's numeric forecast + RAG-retrieved news and
asks an LLM (Mistral, via langchain-mistralai) to produce:
  1. a sentiment/risk extraction over the retrieved news
  2. a buy/sell/hold call with a plain-English rationale that reconciles the
     quantitative forecast with the qualitative news context
"""
from __future__ import annotations
import json
from app import config

SYSTEM_PROMPT = """You are a financial analyst assistant. You are given:
1. A quantitative 7-day price forecast produced by an LSTM model (median, 10th/90th percentile band).
2. A set of retrieved recent news snippets about the company and macro environment.

Your job: produce a JSON object with exactly these keys:
- "sentiment": one of "positive", "negative", "mixed", "neutral"
- "key_risks": array of up to 4 short strings, each a specific risk factor mentioned or implied in the news
- "key_catalysts": array of up to 4 short strings, positive drivers mentioned or implied in the news
- "recommendation": one of "buy", "sell", "hold"
- "confidence": one of "low", "medium", "high"
- "rationale": a 3-5 sentence plain-English explanation that explicitly connects the numeric
  forecast direction/magnitude to the qualitative news themes. Be balanced — mention
  counterpoints, not just the case for your recommendation. Do not overstate certainty;
  this is a forecast, not a guarantee.

Respond with ONLY the JSON object, no preamble, no markdown code fences."""


def _build_user_prompt(ticker: str, forecast: dict, news: list[dict]) -> str:
    news_block = "\n\n".join(
        f"- [{n['date']}] {n['headline']}\n  {n['text']}" for n in news
    )
    return f"""Ticker: {ticker}
Last close: {forecast['last_close']}
7-day forecast (median path): {[f["median"] for f in forecast['forecast']]}
10th percentile path: {[f["low_p10"] for f in forecast['forecast']]}
90th percentile path: {[f["high_p90"] for f in forecast['forecast']]}
Expected return over 7 days: {forecast['expected_return_pct']}%

Recent news and macro context:
{news_block}

Produce the JSON object as instructed."""


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    if not config.MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY not set. Export it, or swap _call_llm() for a "
            "different provider (see module docstring)."
        )
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatMistralAI(
        model=config.MISTRAL_MODEL,
        api_key=config.MISTRAL_API_KEY,
        temperature=0.2,
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return response.content


def synthesize(ticker: str, forecast: dict, news: list[dict]) -> dict:
    user_prompt = _build_user_prompt(ticker, forecast, news)
    raw = _call_llm(SYSTEM_PROMPT, user_prompt)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "sentiment": "unknown",
            "key_risks": [],
            "key_catalysts": [],
            "recommendation": "hold",
            "confidence": "low",
            "rationale": f"LLM response could not be parsed as JSON. Raw response: {raw[:500]}",
        }


if __name__ == "__main__":
    from app.models.predict import forecast as get_forecast
    from app.rag.retriever import retrieve_for_ticker

    ticker = "NVDA"
    fc = get_forecast(ticker)
    news = retrieve_for_ticker(ticker, k=4)
    result = synthesize(ticker, fc, news)
    print(json.dumps(result, indent=2))
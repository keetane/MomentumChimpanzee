import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

JST = ZoneInfo("Asia/Tokyo")


@dataclass
class Pick:
    ticker: str
    company: str
    reason: str
    sentiment: str
    buy_limit: float
    stop_loss: float


def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/app.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["company"] = df["company"].astype(str).str.strip()
    return df


def to_jp_ticker(ticker: str) -> str:
    if ticker.endswith(".T"):
        return ticker
    return f"{ticker}.T"


def rsi(series: pd.Series, period: int = 14) -> float:
    if series is None or len(series) < period + 1:
        return float("nan")
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down
    rsi_val = 100 - (100 / (1 + rs))
    return float(rsi_val.iloc[-1])


def atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 1:
        return float("nan")
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(atr_val)


def compute_metrics(prices: pd.DataFrame, ticker: str) -> dict:
    df = prices[ticker] if isinstance(prices.columns, pd.MultiIndex) else prices
    df = df.dropna()
    if df.empty:
        return {}

    close = df["Close"]
    last_close = float(close.iloc[-1])
    ma5 = float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else float("nan")
    ma25 = float(close.rolling(25).mean().iloc[-1]) if len(close) >= 25 else float("nan")
    avg_vol20 = float(df["Volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float("nan")
    rsi14 = rsi(close, 14)
    atr14 = atr(df, 14)
    vol_score = float(atr14 / last_close) if last_close > 0 and not np.isnan(atr14) else float("nan")
    trend = "up" if not np.isnan(ma5) and not np.isnan(ma25) and ma5 > ma25 else "down"

    return {
        "last_close": last_close,
        "ma5": ma5,
        "ma25": ma25,
        "avg_vol20": avg_vol20,
        "rsi14": rsi14,
        "atr14": atr14,
        "vol_score": vol_score,
        "trend": trend,
    }


def fetch_jp_market(universe: pd.DataFrame) -> list[dict]:
    tickers = [to_jp_ticker(t) for t in universe["ticker"].tolist()]
    prices = yf.download(tickers=tickers, period="6mo", interval="1d", group_by="ticker", auto_adjust=False, threads=True)

    result = []
    for _, row in universe.iterrows():
        ticker = to_jp_ticker(row["ticker"])
        metrics = compute_metrics(prices, ticker)
        if not metrics:
            continue
        result.append(
            {
                "ticker": ticker,
                "company": row["company"],
                **metrics,
            }
        )
    return result


def pct_change(series: pd.Series, periods: int) -> float:
    if series is None or len(series) <= periods:
        return float("nan")
    return float((series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100)


def fetch_us_market() -> dict:
    indices = {
        "dow": "^DJI",
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "sox": "^SOX",
        "oil": "CL=F",
        "usd_jpy": "USDJPY=X",
    }
    mag7 = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]

    tickers = list(indices.values()) + mag7
    prices = yf.download(tickers=tickers, period="20d", interval="1d", group_by="ticker", auto_adjust=False, threads=True)

    def last_close(t: str) -> pd.Series:
        df = prices[t]
        return df["Close"].dropna()

    summary = {}
    for name, t in indices.items():
        close = last_close(t)
        summary[name] = {
            "ticker": t,
            "last_close": float(close.iloc[-1]) if len(close) else float("nan"),
            "chg_1d_pct": pct_change(close, 1),
            "chg_5d_pct": pct_change(close, 5),
        }

    mag7_changes = []
    for t in mag7:
        close = last_close(t)
        mag7_changes.append(
            {
                "ticker": t,
                "chg_1d_pct": pct_change(close, 1),
                "chg_5d_pct": pct_change(close, 5),
            }
        )

    mag7_1d = np.nanmean([x["chg_1d_pct"] for x in mag7_changes])
    mag7_5d = np.nanmean([x["chg_5d_pct"] for x in mag7_changes])

    summary["mag7"] = {
        "tickers": mag7,
        "avg_chg_1d_pct": float(mag7_1d),
        "avg_chg_5d_pct": float(mag7_5d),
    }

    return summary


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def call_grok(model: str, system: str, user: str) -> str:
    client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=[
            {"type": "x_search"},
            {"type": "web_search"},
        ],
    )

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    # フォールバック
    try:
        return response.output[0].content[0].text
    except Exception:
        return str(response)


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("JSONが見つかりません")
    return json.loads(match.group(0))


def calc_levels(last_close: float, atr14: float) -> tuple[float, float]:
    if last_close <= 0 or np.isnan(last_close):
        return float("nan"), float("nan")

    if np.isnan(atr14) or atr14 <= 0:
        buy = last_close * 0.99
        stop = buy * 0.97
    else:
        buy = last_close - (0.3 * atr14)
        stop = buy - (0.7 * atr14)

    return round(buy, 1), round(stop, 1)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def send_discord(message: str) -> None:
    webhook = os.environ["DISCORD_WEBHOOK_URL"]
    payload = {"content": message}
    resp = requests.post(webhook, json=payload, timeout=30)
    if resp.status_code >= 300:
        raise RuntimeError(f"Discord送信失敗: {resp.status_code} {resp.text}")


def build_prompt(jp_metrics: list[dict], us_summary: dict, max_picks: int) -> tuple[str, str]:
    system = (
        "あなたは日本株デイトレ特化の市場分析AIです。\n"
        "目的: 日本Xのセンチメント(60%)と米国市場(40%)を統合し、最大10銘柄の注目株を抽出する。\n"
        "必須: 客観的・具体的、皮肉/ジョーク補正、重要投稿抽出、後場跨ぎ禁止を前提に判断。\n"
        "ツール: x_searchで日本株関連投稿(過去48h)を取得し、web_searchで米国イベントと市場反応を補足。\n"
        "出力はJSONのみ。"
    )

    user = {
        "task": "日本株注目銘柄の抽出",
        "as_of_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
        "max_picks": max_picks,
        "weights": {
            "jp_x": 0.6,
            "us_market": 0.4,
        },
        "us_market_summary": us_summary,
        "jp_universe_metrics": jp_metrics,
        "instructions": {
            "x_search_query": "日本株 OR 個別株 OR 仕手株 OR 決算 OR 上方修正 OR 下方修正 OR 半導体 OR 防衛 OR 自動車",
            "x_search_window_hours": 48,
            "output_json_schema": {
                "sentiment_summary": {
                    "jp_x": {"positive_ratio": "float", "negative_ratio": "float", "key_themes": "list"},
                    "us_market": {"key_events": "list", "market_reaction": "list"},
                    "overall": {"tone": "string", "impact_summary": "string"},
                },
                "important_posts": [
                    {"summary": "string", "reason": "string"}
                ],
                "picks": [
                    {
                        "ticker": "string",
                        "company": "string",
                        "sentiment": "string",
                        "reason": "string"
                    }
                ],
            },
        },
    }

    return system, json.dumps(user, ensure_ascii=False)


def format_report(picks: list[Pick], sentiment_summary: dict, important_posts: list[dict]) -> str:
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    lines = [f"注目銘柄レポート ({now})", ""]

    if sentiment_summary:
        lines.append("全体センチメント")
        overall = sentiment_summary.get("overall", {})
        lines.append(f"- トーン: {overall.get('tone', 'N/A')}")
        lines.append(f"- 影響要約: {overall.get('impact_summary', 'N/A')}")
        lines.append("")

    if important_posts:
        lines.append("重要投稿ハイライト")
        for p in important_posts[:5]:
            lines.append(f"- {p.get('summary', '')} ({p.get('reason', '')})")
        lines.append("")

    lines.append("注目銘柄")
    for p in picks:
        lines.append(f"- {p.company} ({p.ticker})")
        lines.append(f"  買い指値: {p.buy_limit} / 逆指値: {p.stop_loss}")
        lines.append(f"  理由: {p.reason}")
        lines.append(f"  センチメント: {p.sentiment}")

    lines.append("")
    lines.append("注意: デイトレ専用。後場跨ぎ厳禁。逆指値は必須。")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    setup_logging()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEYが未設定です")

    model = os.getenv("XAI_MODEL", "grok-4-1-fast-reasoning")
    max_picks = int(os.getenv("MAX_PICKS", "10"))

    universe = load_universe("data/universe.csv")
    logging.info("日本株ユニバース: %d銘柄", len(universe))

    jp_metrics = fetch_jp_market(universe)
    logging.info("日本株メトリクス取得: %d銘柄", len(jp_metrics))

    us_summary = fetch_us_market()

    system, user = build_prompt(jp_metrics, us_summary, max_picks)
    logging.info("Grok呼び出し開始")
    response_text = call_grok(model, system, user)

    logging.info("Grok応答受信")
    data = extract_json(response_text)

    picks_raw = data.get("picks", [])
    metrics_by_ticker = {m["ticker"]: m for m in jp_metrics}

    picks: list[Pick] = []
    for item in picks_raw:
        ticker = item.get("ticker")
        if not ticker:
            continue
        metrics = metrics_by_ticker.get(ticker)
        if not metrics:
            continue
        buy, stop = calc_levels(metrics["last_close"], metrics["atr14"])
        picks.append(
            Pick(
                ticker=ticker,
                company=item.get("company", ticker),
                reason=item.get("reason", ""),
                sentiment=item.get("sentiment", ""),
                buy_limit=buy,
                stop_loss=stop,
            )
        )

    sentiment_summary = data.get("sentiment_summary", {})
    important_posts = data.get("important_posts", [])

    report = format_report(picks, sentiment_summary, important_posts)
    os.makedirs("data", exist_ok=True)
    with open("data/last_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    logging.info("Discord送信開始")
    send_discord(report)
    logging.info("Discord送信完了")


if __name__ == "__main__":
    main()

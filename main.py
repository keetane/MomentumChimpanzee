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
    side: str
    entry: float
    take_profit: float
    stop_loss: float
    ma5: float
    ma25: float
    recent_high_20: float


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
    recent_high_20 = float(df["High"].rolling(20).max().iloc[-1]) if len(df) >= 20 else float("nan")
    avg_vol20 = float(df["Volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float("nan")
    rsi14 = rsi(close, 14)
    atr14 = atr(df, 14)
    vol_score = float(atr14 / last_close) if last_close > 0 and not np.isnan(atr14) else float("nan")
    trend = "up" if not np.isnan(ma5) and not np.isnan(ma25) and ma5 > ma25 else "down"

    return {
        "last_close": last_close,
        "ma5": ma5,
        "ma25": ma25,
        "recent_high_20": recent_high_20,
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


def fetch_nikkei_futures_6jst() -> dict | None:
    # 日経先物（CME）を1時間足で取得し、当日6:00 JSTの価格を参照する
    ticker = "NK=F"
    now_jst = datetime.now(JST)
    start = (now_jst - timedelta(days=2)).strftime("%Y-%m-%d")
    df = yf.download(tickers=ticker, start=start, interval="1h", auto_adjust=False, progress=False)
    if df.empty:
        return None

    idx = df.index
    # yfinanceはUTCのことが多いのでJSTに変換
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(JST)
    df = df.copy()
    df.index = idx

    target_date = now_jst.date()
    candidate = df[(df.index.date == target_date) & (df.index.hour == 6)]
    if candidate.empty:
        # 6:00が取れなければスキップ
        return None

    row = candidate.iloc[-1]
    ts = candidate.index[-1].strftime("%Y-%m-%d %H:%M")
    return {"ticker": ticker, "price": float(row["Close"]), "time_jst": ts}


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


def calc_levels(last_close: float, atr14: float, side: str, r_multiple: float) -> tuple[float, float, float]:
    if last_close <= 0 or np.isnan(last_close):
        return float("nan"), float("nan"), float("nan")

    if np.isnan(atr14) or atr14 <= 0:
        if side == "short":
            entry = last_close * 1.01
            stop = entry * 1.03
        else:
            entry = last_close * 0.99
            stop = entry * 0.97
    else:
        if side == "short":
            entry = last_close + (0.3 * atr14)
            stop = entry + (0.7 * atr14)
        else:
            entry = last_close - (0.3 * atr14)
            stop = entry - (0.7 * atr14)

    risk = abs(entry - stop)
    take = entry + (r_multiple * risk) if side == "long" else entry - (r_multiple * risk)

    return round(entry, 1), round(take, 1), round(stop, 1)


def build_tl_scan_prompt(max_count: int) -> tuple[str, str]:
    system = (
        "あなたは日本株デイトレ特化の情報収集AIです。\\n"
        "目的: Xのタイムライン(過去48h)から話題性が高い日本株銘柄を抽出する。\\n"
        "必須: 皮肉・ジョーク補正。話題量と感情の勢いを評価し、ランキングで返す。\\n"
        "出力はJSONのみ。"
    )

    user = {
        "task": "日本株の話題性ランキング抽出",
        "max_count": max_count,
        "x_search_query": "日本株 OR 個別株 OR 仕手株 OR 決算 OR 上方修正 OR 下方修正 OR 半導体 OR 防衛 OR 自動車",
        "x_search_window_hours": 48,
        "emotion_weights": {"joy": 0.3, "surprise": 0.4, "fear": 0.2, "sadness": 0.1},
        "score_weights": {"mention": 0.4, "emotion": 0.5, "quality": 0.1},
        "output_json_schema": {
            "candidates": [
                {
                    "ticker": "string",
                    "company": "string",
                    "mention_count": "int",
                    "emotion_intensity": "float",
                    "heat_score": "float",
                }
            ]
        },
    }
    return system, json.dumps(user, ensure_ascii=False)


def split_for_discord(message: str, max_len: int = 1900) -> list[str]:
    # Discordの2000文字制限を超えないように分割
    chunks: list[str] = []
    buf: list[str] = []
    current = 0
    for line in message.splitlines():
        line_len = len(line) + 1
        if current + line_len > max_len and buf:
            chunks.append("\n".join(buf))
            buf = [line]
            current = line_len
        else:
            buf.append(line)
            current += line_len
    if buf:
        chunks.append("\n".join(buf))
    return chunks


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def send_discord(message: str) -> None:
    webhook = os.environ["DISCORD_WEBHOOK_URL"]
    for part in split_for_discord(message):
        payload = {"content": part}
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
        "momentum_score_weights": {"mention": 0.4, "emotion": 0.5, "quality": 0.1},
        "emotion_weights": {"joy": 0.3, "surprise": 0.4, "fear": 0.2, "sadness": 0.1},
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
                        "reason": "string",
                        "side": "long_or_short"
                    }
                ],
                "tl_momentum_ranking": [
                    {
                        "ticker": "string",
                        "company": "string",
                        "mention_count": "int",
                        "emotion_intensity": "float",
                        "heat_score": "float"
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
        lines.append(f"  方向: {p.side}")
        lines.append(f"  エントリー指値: {p.entry} / 利確目標: {p.take_profit} / 逆指値: {p.stop_loss}")
        lines.append(f"  理由: {p.reason}")
        lines.append(f"  センチメント: {p.sentiment}")
        lines.append(f"  MA5: {p.ma5} / MA25: {p.ma25} / 直近20日高値: {p.recent_high_20}")

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
    r_multiple = float(os.getenv("R_MULTIPLE", "1.5"))

    universe = load_universe("data/universe.csv")
    base_universe = universe.copy()
    base_tickers = set(to_jp_ticker(t) for t in base_universe["ticker"].tolist())
    logging.info("日本株ユニバース: %d銘柄", len(universe))

    extra_max = int(os.getenv("EXTRA_TL_TICKERS_MAX", "10"))
    tl_system, tl_user = build_tl_scan_prompt(extra_max)
    logging.info("TLスキャン開始")
    tl_text = call_grok(model, tl_system, tl_user)
    tl_data = extract_json(tl_text)
    candidates = tl_data.get("candidates", [])

    extra_rows = []
    for c in candidates:
        ticker = str(c.get("ticker", "")).strip()
        if not ticker:
            continue
        extra_rows.append({"company": c.get("company", ticker), "ticker": ticker})

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        universe = pd.concat([universe, extra_df], ignore_index=True).drop_duplicates(subset=["ticker"])
        logging.info("TL候補を追加: %d銘柄", len(extra_df))

    jp_metrics = fetch_jp_market(universe)
    logging.info("日本株メトリクス取得: %d銘柄", len(jp_metrics))

    us_summary = fetch_us_market()
    nikkei_6 = fetch_nikkei_futures_6jst()
    if nikkei_6 is not None:
        us_summary["nikkei_futures_6jst"] = nikkei_6

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
        if not ticker.endswith(".T"):
            ticker = f"{ticker}.T"
        metrics = metrics_by_ticker.get(ticker)
        if not metrics:
            continue
        side = str(item.get("side", "long")).lower()
        side = "short" if side == "short" else "long"
        entry, take, stop = calc_levels(metrics["last_close"], metrics["atr14"], side, r_multiple)
        picks.append(
            Pick(
                ticker=ticker,
                company=item.get("company", ticker),
                reason=item.get("reason", ""),
                sentiment=item.get("sentiment", ""),
                side=side,
                entry=entry,
                take_profit=take,
                stop_loss=stop,
                ma5=metrics.get("ma5", float("nan")),
                ma25=metrics.get("ma25", float("nan")),
                recent_high_20=metrics.get("recent_high_20", float("nan")),
            )
        )

    # 固定リスト外の銘柄を先に出す
    picks.sort(key=lambda p: p.ticker in base_tickers)

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

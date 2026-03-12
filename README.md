# MomentumChimpanzee

日本株デイトレ向けの注目銘柄を、Xのセンチメントと米国市場データを組み合わせて抽出し、Discordに通知する自動バッチです。

## 概要
- Grok API（x_search / web_search）で日本株関連投稿と米国イベントを収集
- 日本X 60% / 米国市場 40% でセンチメントを統合
- yfinanceでRSI・移動平均・出来高・ATRを計算
- 最大10銘柄を抽出し、買い指値・逆指値・理由を出力
- Discord Webhookに通知

## セットアップ
1. `requirements.txt` をインストール
2. `.env` を作成（`.env.example` を参照）
3. `python main.py` を実行

## GitHub Actions
`.github/workflows/daily.yml` を参照してください。

## 注意
- デイトレ専用（後場跨ぎ厳禁）
- APIキーは必ず `.env` に保存し、Gitに含めないでください

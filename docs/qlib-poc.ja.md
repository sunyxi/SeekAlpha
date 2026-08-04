# Qlib リサーチアダプター POC

この POC は、SeekAlpha の既存 `DailyPanel` を決定論的な Qlib 互換
pandas DataFrame に変換します。インデックスは `(datetime, instrument)`、
列は `$open`、`$high`、`$low`、`$close`、`$volume` です。

アーキテクチャ制約は
[ADR-006](adr/ADR-006-optional-qlib-adapter.md) に定義されています。

## CLI Usage

オプションのランタイムをインストールします。

```bash
pip install -e ".[qlib]"
```

既存の分足キャッシュから、再ダウンロードせずに出力します。

```bash
python3 scripts/qlib_poc.py \
  --cache-dir data \
  --start 2021-01-04 \
  --end 2026-06-30 \
  --output derived/qlib-daily.csv \
  --verify-qlib
```

Qlib 本体の確認が不要な場合は `--verify-qlib` を省略できます。

## Operations

- `data/` は読み取り専用の入力として扱います。CLI は既存の
  `*_1min.csv.gz` のみを読み、Alpaca などへ接続しません。
- 出力先は `data/` の外、例えば `derived/` にします。
- 出力は create-only です。再実行時は新しいパスを指定し、コマンド、
  Git commit、入力キャッシュと出力のハッシュを記録します。
- 環境変更後は Qlib 関連の unit、CLI fixture、runtime integration test を
  実行します。
- Apple Silicon では、インストールエラーの内容に応じて OpenMP
  ランタイムが必要になる場合があります。

## Limitations

- この POC はデータ境界だけを検証します。Alpha158、モデル学習、
  パラメータ調整、収益改善の主張は含みません。
- 入力は IEX 分足キャッシュ由来の日次 OHLCV であり、統合板情報や
  全市場出来高ではありません。
- `build_panel` は指定日付を適用する前に対象キャッシュファイル全体を
  走査します。そのため、複数年キャッシュから 1 日だけ出力する場合も
  数分かかることがあります。
- 観測開始前の欠損値は NaN のままです。既存の forward-fill はそのまま
  保持されます。
- Qlib のバックテストは、凍結済み ORB ライフサイクル、コスト条件、
  nested walk-forward、独立した執行検証を置き換えません。
- `pyqlib` 未導入時の runtime fixture は skipped と報告し、passed と
  扱ってはいけません。
- pyqlib 0.9.7 では `qlib.constant` から numpy `Timedelta` の非推奨警告が
  3 件出ます。現在の fixture は通過しますが、将来の numpy では Qlib の
  更新が必要になる可能性があります。

## Rollback

Qlib adapter、CLI、`qlib` optional dependency、関連テスト、ADR-006、
三言語文書を削除します。既存キャッシュ、ORB core、walk-forward report、
凍結済み factor list、ブローカー向け資産の移行や巻き戻しは不要です。

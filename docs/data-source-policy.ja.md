# データソースポリシー

## 決定

暫定 primary は Databento、fallback は Alpaca です。状態は
`selected_with_blockers` であり、ダウンロード許可ではありません。Source of
Truth は `src/orb/data_source_policy.json` です。

## CLI 使用方法

```bash
python3 scripts/validate_data_source_policy.py
```

## 運用

bulk download 前に契約、non-display research 権利、保存期間、corporate action、
sector history の範囲と費用を確認します。認証情報は環境変数からのみ読み、raw
data や認証情報を Git に保存しません。

membership は `instrument_id` と半開区間で再構築します。ticker は日付範囲内の
alias にすぎず、現行 constituent を過去の membership として使用しません。
該当日の delisted instrument の履歴は保持します。

## 制限

Databento の sector history entitlement と最終契約は未設定です。Alpaca は bars、
quotes、corporate actions の fallback であり、PIT membership の証明ではありません。
この Issue ではデータ取得や認証を実行しません。

## ロールバック

取得前なら新しい branch で policy と ADR を修正します。取得後は manifest や
membership を書き換えず、新しい policy ID を作成して関連実験を無効化します。

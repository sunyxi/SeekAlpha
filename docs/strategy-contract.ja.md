# 戦略コントラクト

## 目的

実装前に反証可能な仮説を登録し、判定レポートを仕様、研究プロトコル、
データスナップショット、コードリビジョンへ固定します。Source of Truth は
`src/orb/strategy_spec_template.json` です。

## CLI 使用方法

```bash
python3 scripts/validate_strategy_contract.py \
  --spec path/to/spec.json --report path/to/report.json \
  --summary-output reports/strategy-summary.md
```

判定状態は `Candidate`、`No-Go`、`Exploratory`、`Invalid` のみ許可します。
概要は create-only で生成し、JSON から再生成します。

## 運用

実装前に universe、feature、label の時点、保有期間、コスト、探索空間、
protocol hash、data manifest hash、予算を固定します。各レポートには spec
hash を記録し、gate 状態は `passed`、`failed`、`not-run`、`skipped` のいずれか
で記録します。

## 制限

収益性評価、データ取得、retention 読み取り、モデル実行は行いません。コード
コミットは空でない識別子として要求しますが、ローカル検証器は取得・検証しません。

## ロールバック

結果前なら feature branch を revert して仕様を修正します。公開済みレポートは
上書きせず、新しい spec または experiment report を作成し、無効化理由を記録して
概要を再生成します。

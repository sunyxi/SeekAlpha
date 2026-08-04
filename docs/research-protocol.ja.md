# リサーチプロトコル

## 目的

Issue #53 では、新しい戦略実験の前に研究境界を固定します。Source of
Truth は `src/orb/research_protocol.json` です。結果を確認した後に値を
変更したり、戦略コードへコピーしたりしてはいけません。

## 固定された管理項目

- 開発期間: 2021-01-04 から 2024-12-31。
- Outer test: 2025-01-01 から 2026-06-30。
- Retention: 2026-07-01 から 2026-12-31、利用可能日は 2027-01-01。
- 戦略ファミリーごとに最大3実験、全体で最大12実験。
- 1実験あたり最大192パラメータ試行、50モデル試行。
- 252日学習、63日検証、63日 outer test、63日ステップ、20日 purge、
  5日 embargo。
- コストは片道 0、2.5、5.0 bps。
- 乱数を使う処理は宣言済み seed を記録します。

## CLI 使用方法

```bash
python3 scripts/validate_research_protocol.py --json
```

retention 評価器は `RetentionLedger.read_once(experiment_id, loader)` を
使用します。loader の失敗でもアクセス権は消費されます。

## 運用

プロトコル、データ manifest、search-space hash、seed、experiment ID を
レポートへ保存し、ledger を改変不能な研究ストレージに保管します。
`retention_available_after` より前に retention を読みません。

## 制限

これはデータ品質や分散監査サービスを提供せず、ledger の削除権限を持つ
利用者を防げません。後続のデータおよび実験ハーネス Issue で扱います。

## ロールバック

Accepted の JSON を直接編集しません。結果や retention 利用前なら feature
branch を revert します。その後は新しい protocol ID と experiment ID を
作り、無効化理由を記録して開発期間から再実行します。消費済み marker は
削除しません。

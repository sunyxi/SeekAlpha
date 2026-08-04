# 不変リサーチキャッシュ

## CLI 使用方法

```bash
python3 scripts/validate_snapshot.py \
  --cache-root research-cache --snapshot-id daily-panel-2026-06-30
```

## 運用

provider、request、期間、symbol、schema、chunk 数、loader を指定して
`ImmutableResearchCache.ingest_partition` を使用します。成功した chunk ごとに
deterministic な `.partial` checkpoint を保存します。完了後にだけ snapshot を
publish し、ID と manifest hash を experiment report に記録します。

## 制限

ローカル filesystem の実装であり、分散 lock や cloud retention は提供しません。
provider data の取得・ライセンス検証・異なる schema の統合は行いません。既存 ORB
cache updater は互換用に分離され、新しい snapshot 作成には使いません。

## ロールバック

experiment が参照する partition を削除・上書きしません。不正な取り込みは snapshot
を無効化し、修正後に新しい partition/snapshot ID を発行します。checkpoint は診断用に
保持しますが、秘密情報を含む場合は secret rotation 手順に従います。

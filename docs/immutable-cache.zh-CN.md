# 不可变研究缓存

## CLI 使用

```bash
python3 scripts/validate_snapshot.py \
  --cache-root research-cache --snapshot-id daily-panel-2026-06-30
```

## 运维

通过 `ImmutableResearchCache.ingest_partition` 提供 provider、request、时间范围、
symbols、schema、chunk 数和 loader。每个成功 chunk 更新 deterministic `.partial`
checkpoint；完成所有分区后才能 publish snapshot，并把 snapshot ID 和 manifest hash
记录到实验报告。

## 限制

本 Issue 只提供本地文件系统存储，不提供分布式锁或云端 retention；不下载供应商数据、
不验证授权，也不合并不兼容 schema。旧 ORB cache updater 保持兼容隔离，不得用于创建
新策略 snapshot。

## 回滚

不得删除或覆盖实验引用的 partition。错误 ingestion 应标记 snapshot 失效，修正后创建
新的 partition/snapshot ID。checkpoint 应保留用于诊断；若包含秘密信息，按 secret
rotation 流程处理。

# 研究协议

## 目的

Issue #53 在新策略实验前冻结研究边界。唯一 Source of Truth 是
`src/orb/research_protocol.json`，结果出现后不得修改或复制到策略代码中。

## 冻结控制项

- 开发期：2021-01-04 至 2024-12-31。
- Outer test：2025-01-01 至 2026-06-30。
- Retention：2026-07-01 至 2026-12-31，2027-01-01 后才可读取。
- 每个策略族最多 3 个实验，总计最多 12 个实验。
- 每个实验最多 192 次参数试验、50 次模型试验。
- 252 天训练、63 天验证、63 天 outer test、63 天步长、20 天 purge、
  5 天 embargo。
- 成本场景为每边 0、2.5、5.0 bps。
- 所有随机过程必须记录预先声明的 seed。

## CLI 使用

```bash
python3 scripts/validate_research_protocol.py --json
```

Retention 评估器必须调用 `RetentionLedger.read_once(experiment_id, loader)`。
ledger 在 loader 执行前预约访问；即使 loader 失败，该访问仍视为已消耗。

## 运维

运行实验前校验协议，并在报告记录 protocol hash、数据 manifest hash、
search-space hash、seed 和 experiment ID。ledger 必须保存在不可变研究存储
中；在 `retention_available_after` 之前禁止读取 retention。

## 限制

协议不负责证明数据质量，也不提供分布式审计服务；拥有文件系统写权限的
用户仍可能删除 marker。这些内容由后续数据和实验框架 Issue 处理。

## 回滚

不得原地编辑已接受的 JSON。若尚未产生结果或读取 retention，可回退 feature
branch。之后必须创建新的 protocol ID 和 experiment ID，记录失效原因，并从
开发期重新运行；不得删除已消耗的 ledger marker。

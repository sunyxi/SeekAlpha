# 策略契约

## 目的

在编码前登记可证伪的策略假设，并将决策报告绑定到准确的规格、研究协议、
数据快照和代码版本。字段 Source of Truth 是
`src/orb/strategy_spec_template.json`。

## CLI 使用

```bash
python3 scripts/validate_strategy_contract.py \
  --spec path/to/spec.json --report path/to/report.json \
  --summary-output reports/strategy-summary.md
```

决策状态只允许 `Candidate`、`No-Go`、`Exploratory`、`Invalid`。摘要只能
create-only 生成；源 JSON 变化后必须重新生成，不能手工修改生成文件。

## 运维

实现策略前冻结 universe、feature、label 时点、持仓周期、成本、搜索空间、
protocol hash、data manifest hash 和预算。每份报告记录 spec hash，所有 gate
使用 `passed`、`failed`、`not-run`、`skipped` 之一。

## 限制

本契约不评估收益率、不下载数据、不读取 retention，也不运行模型。代码 commit
只要求是非空标识符，本地校验器不会拉取或验证该提交。

## 回滚

产生结果前可回退 feature branch 并修正规格。报告发布后不得覆盖；应创建新的
spec 或 experiment report，记录失效原因，再生成新的摘要。

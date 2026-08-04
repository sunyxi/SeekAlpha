# 数据源政策

## 决策

暂定 Databento 为 primary，Alpaca 为 fallback。状态是
`selected_with_blockers`，不是下载授权。Source of Truth 是
`src/orb/data_source_policy.json`。

## CLI 使用

```bash
python3 scripts/validate_data_source_policy.py
```

## 运维

bulk download 前必须确认供应商合同、non-display research 权限、保留权利、
corporate action 覆盖、历史 sector history 覆盖和成本。凭据只能从环境变量读取，
不得提交 raw data 或凭据。

成员资格使用 `instrument_id` 与半开 listing interval 重建；ticker 只能作为日期
限定的 alias。按研究日期保留有效的退市标的，禁止用当前 constituents 代替历史成员。

## 限制

Databento 的 sector history entitlement 和最终合同尚未配置。Alpaca 可作为 bars、
quotes、corporate actions fallback，但不能证明 point-in-time membership。本 Issue
不执行下载或认证。

## 回滚

下载前在新分支修正 policy 和 ADR。下载后不得改写 manifest 或重新解释成员资格；
应创建新的 policy ID 并使受影响实验失效。不要用删除 Git 历史替代凭据轮换流程。

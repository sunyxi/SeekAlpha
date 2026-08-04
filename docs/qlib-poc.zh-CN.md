# Qlib 研究适配器 POC

本 POC 将 SeekAlpha 现有 `DailyPanel` 转换为确定性的 Qlib 兼容 pandas
DataFrame。索引为 `(datetime, instrument)`，列为 `$open`、`$high`、
`$low`、`$close`、`$volume`。

架构约束见 [ADR-006](adr/ADR-006-optional-qlib-adapter.md)。

## CLI Usage

安装可选运行时：

```bash
pip install -e ".[qlib]"
```

只使用现有分钟缓存导出，不重新下载数据：

```bash
python3 scripts/qlib_poc.py \
  --cache-dir data \
  --start 2021-01-04 \
  --end 2026-06-30 \
  --output derived/qlib-daily.csv \
  --verify-qlib
```

只需要表格导出时可以省略 `--verify-qlib`。

## Operations

- 将 `data/` 视为只读输入。CLI 只读取现有 `*_1min.csv.gz`，不会访问
  Alpaca 或其他数据源。
- 输出写入 `data/` 之外，例如 `derived/`。
- 输出遵循 create-only。每次运行使用新路径，并保留命令、Git commit、
  源缓存哈希和输出哈希。
- 环境变化后运行 Qlib 单元测试、CLI Fixture 和运行时集成测试。
- Apple Silicon 只有在安装错误明确提示缺少 OpenMP 时才安装对应运行库。

## Limitations

- 本 POC 只验证数据集成边界，不运行 Alpha158、不训练模型、不调参，
  也不声明收益改善。
- 输入是现有 IEX 分钟缓存聚合的日线 OHLCV，不包含全市场订单簿或
  consolidated volume。
- `build_panel` 当前会先扫描每个选中缓存文件的完整内容，再应用日期范围；
  因此即使只从多年缓存导出一天，也可能需要数分钟。
- 首次观测前的缺失值继续保留为 NaN；`DailyPanel` 已有的后续前向填充
  原样保留。
- Qlib 回测不能替代已冻结的 ORB 持仓生命周期、成本场景、nested
  walk-forward 和独立执行验证。
- 未安装 `pyqlib` 时，运行时 Fixture 必须报告 skipped，不能报告 passed。
- pyqlib 0.9.7 当前会从 `qlib.constant` 产生 3 条 numpy `Timedelta` 弃用
  警告。Fixture 仍然通过，但未来 numpy 版本可能要求升级 Qlib。

## Rollback

删除 Qlib adapter、CLI、`qlib` optional dependency、相关测试、ADR-006
和三语 POC 文档即可。现有缓存、ORB core、walk-forward 报告、冻结因子
列表和券商侧资产不需要迁移或回滚。

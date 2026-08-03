# SeekAlpha — ORB 研究与回测工具

一次数据扫描模拟全部 192 个冻结候选，离线完成 nested walk-forward、
成本敏感性与 Candidate / No-Go 判定。不依赖 QuantConnect。

## 结构

```text
core/orb_core.py        券商无关 ORB 核心：特征、信号、192 候选生命周期模拟
scripts/local_pump.py   数据泵：Alpaca 免费 IEX 分钟数据 -> 5 分钟合成 -> 核心
scripts/wf_select.py    nested walk-forward + 成本场景 + 报告（create-only）
tests/test_orb_core.py  确定性 Fixture 测试（无网络、无账户）
docs/ARCHITECTURE.md    系统架构与关键不变量
docs/ROADMAP.md         里程碑路线图（M0–M5）
```

## 安装

```bash
pip install alpaca-py            # 唯一的第三方依赖；核心与判定脚本零依赖
python3 -m unittest discover -s tests -v   # 先本地验证核心
```

在 alpaca.markets 注册免费 paper 账户（不需要入金），拿到 API Key：

```bash
export ALPACA_API_KEY=你的key
export ALPACA_SECRET_KEY=你的secret
```

## 运行

```bash
# 1) 下载数据(自动缓存到 data/，重跑不重复下载)并模拟全部候选
python3 scripts/local_pump.py --start 2021-01-04 --end 2026-06-30 \
  --cache-dir data --out-dir runs/orb045

# 2) walk-forward 判定
python3 scripts/wf_select.py --trades-dir runs/orb045 \
  --report-output reports/orb045-wf.json
```

## 预冻结的门槛（看到结果前不得修改）

- Fold 内选择：train 交易数 >= 30、train Sharpe >= 0、train PF >= 1.0、
  validation（train 后 20%）净 PnL > 0。
- 最终 Candidate（baseline 2.5 bps/side outer test 聚合）：
  交易数 >= 100、Sharpe > 0.5、PF >= 1.10、mean net bps > 0。
- 任一不满足 -> No-Go。

## 已知限制（报告使用时必须声明）

- IEX 数据只含单一交易所成交量；RVOL 因分子分母同源仍然有效。
- adjustment="all" 把 split/dividend 折入价格。
- 固定 31 只股票池存在幸存者偏差。
- Candidate 只是继续研究的许可，不是交易许可。

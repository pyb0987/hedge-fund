# Paper Trading Validation Framework

## Schedule

- **Frequency**: 1 cycle/day (21:00 KST)
- **Minimum validation period**: 12 weeks (84 cycles)
- **Go/No-Go command**: `uv run python -m hedgefund paper-report`

## Automated Go/No-Go Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Sharpe Ratio | >= 0.5 | Minimum risk-adjusted return |
| Max Drawdown | <= 15% | Small account survival limit |
| Profit Factor | >= 1.1 | Gross profit > gross loss |
| Data sufficiency | >= 10 cycles | Statistical minimum |

## Checkpoint Framework

### Week 2 (14 cycles): System Health

| Check | Pass Condition | Fail Action |
|-------|----------------|-------------|
| Signal generation | Signals produced in >80% of cycles | Debug data providers, strategy logic |
| Signal conversion | Conversion rate > 50% | Check risk limits, position sizing |
| At least 1 trade executed | Any strategy traded | Check rebalancing gates |
| System stability | 0 crashes | Bug fix |

### Week 4 (28 cycles): First Statistical Check

| Check | Pass Condition | Fail Action |
|-------|----------------|-------------|
| Sharpe Ratio | > 0.3 | Review strategy parameters |
| Profit Factor | > 1.0 | Check cost model, signal quality |
| Max Drawdown | < 10% | Warning (reserve judgment) |
| All 3 strategies traded | Each traded at least once | Inspect non-trading strategy |
| Cost accuracy | Paper cost < 1.5x backtest assumption | Adjust cost model if exceeded |

### Week 8 (56 cycles): Second Statistical Check

| Check | Pass Condition | Fail Action |
|-------|----------------|-------------|
| Sharpe Ratio | > 0.3 | Strategy parameter re-tuning |
| Profit Factor | > 1.0 | Cost model or signal quality issue |
| Max Drawdown | < 15% | Reduce position sizing |
| All 3 strategies traded | Yes | Disable non-trading strategy |
| Strategy correlation | < 0.3 pairwise | Redundant strategy — consider replacing |

### Week 12 (84 cycles): Final Go/No-Go

| Check | Pass Condition | Fail Action |
|-------|----------------|-------------|
| Sharpe Ratio | >= 0.5 | **REJECT** — do not deploy |
| Max Drawdown | <= 15% | **REJECT** |
| Profit Factor | >= 1.1 | **REJECT** |
| Data sufficiency | >= 10 cycles | **REJECT** |
| All pass | Yes | **GO** — deploy at quarter-Kelly sizing |

## Manual Monitoring Items

### Signal Fidelity
- Target: conversion rate >= 70%
- If <30% converted: risk system may be over-blocking

### Cost Accuracy
- Compare paper executor slippage (fixed 0.1%) vs actual market spread
- Upbit BTC spread ~0.05-0.15%, ETF spread ~0.01-0.05%
- If paper cost > 1.5x actual: adjust slippage model

### Strategy-Level Decomposition
- Track Sharpe/DD per strategy independently
- If one strategy causes >80% of total loss: disable or re-tune

### Rebalancing Gate Health
- Crypto Momentum: should rebalance every 7 days
- Dual Momentum: should rebalance on monthly rebalance_day
- ETF Mean Reversion: daily z-score check (no gate)
- If 0 trades for 2+ weeks: gate may be stuck

### Known Limitations
- Upbit altcoin data: some small coins return no data (BTC/ETH fine)
- Weekend: no ETF data (Sat/Sun), ETF mean reversion idle
- First few days: 0 trades expected (crypto/dual gated by rebalancing schedule)

## Post-Validation Deployment

After Go decision:
1. Deploy with quarter-Kelly sizing (25% of full Kelly fraction)
2. Start with 50% of total capital, hold rest in cash
3. Monitor daily for first 2 weeks of live trading
4. Scale to full capital after 2 weeks if no anomalies

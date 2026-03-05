# Hedge Fund Project

## Architecture
- Frozen dataclasses for all domain models (immutability)
- Strategy Protocol + ABC pattern for extensibility
- Vectorized backtest (weight matrix approach, not event-loop)
- Progressive drawdown control (5/10/15/20% thresholds)
- Quarter-Kelly position sizing (full Kelly 금지 — 소액 계정 생존 우선)
- Executor Protocol (runtime_checkable) — Paper/Upbit/Alpaca 구현체
- Order에 `price` 필드 = 주문 생성 시점 시장가 (limit_price, filled_price와 구분)

## Testing
- Walk-forward tests: IS_RATIO=0.30, OOS_RATIO=0.10 (기본 60/20은 데이터 부족으로 윈도우 수 확보 불가)
- numpy bool 비교: `is` 대신 `==` 또는 `bool()` 사용 (`np.True_ is True` → False)
- External API providers (upbit, yfinance) 및 거래소 executor (upbit_executor, alpaca_executor) 커버리지 제외 — integration test scope
- Profit factor: MAX_PROFIT_FACTOR=100.0 캡 적용 (inf가 Go/No-Go 통과 방지)
- `backtest_weights()`와 `generate_signals()`는 반드시 동일한 수학 함수 사용 (EP1 data leakage, EP2 z-score 불일치 — 2회 CRITICAL)
- `generate_signals()`에 리밸런싱 게이트 필수: crypto=holding_days, dual=monthly rebalance_day (EP3 daily rebalancing 재발 방지)
- ETF Mean Reversion은 z-score 임계치 기반이므로 리밸런싱 게이트 불필요 (매일 체크가 정상)

## Conventions
- 한국어 설명, 영어 코드/변수명
- Phase 단위 커밋 (feat: implement Phase N — ...)
- Config: YAML + Pydantic 검증
- 거래소: Upbit (crypto KRW), Alpaca (US ETF)

## Paper Trading
- CLI: `uv run python -m hedgefund paper-run [--dry-run] [--verbose]`
- 구동 주기: 일 1회 (매일 21:00 KST 권장)
- 리밸런싱 게이트: crypto=7일, dual_momentum=월간, etf=매일(z-score)
- 상태 저장: `data/paper_state/{upbit,alpaca,strategies}.json`
- Two PaperExecutors: UPBIT (KRW) + ALPACA (USD), 자본 분할 기반

## Key Paths
- Config: `config/settings.yaml`, `config/strategies/*.yaml`
- Core: `src/hedgefund/core/` (models, enums, exceptions, risk_metrics, kelly)
- Strategies: `src/hedgefund/strategies/` (base, crypto_momentum, etf_mean_reversion, dual_momentum, registry)
- Backtest: `src/hedgefund/backtest/` (engine, walk_forward, metrics, deflated_sharpe)
- Risk: `src/hedgefund/risk/` (manager, limits, drawdown, position_sizer)
- Portfolio: `src/hedgefund/portfolio/` (manager, allocator, correlation, rebalancer)
- Execution: `src/hedgefund/execution/` (protocols, cost_model, order_builder, state, executors/)
- App: `src/hedgefund/app.py` (wiring), `cli.py` (typer), `__main__.py`
- Scheduler: `src/hedgefund/scheduler/runner.py`

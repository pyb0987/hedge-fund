# Hedge Fund Project

## Architecture
- Frozen dataclasses for all domain models (immutability)
- Strategy Protocol + ABC pattern for extensibility
- Vectorized backtest (weight matrix approach, not event-loop)
- Progressive drawdown control (5/10/15/20% thresholds)
- Quarter-Kelly position sizing (full Kelly 금지 — 생존 우선)
- Executor Protocol (runtime_checkable) — Paper/Upbit/Alpaca 구현체
- Order에 `price` 필드 = 주문 생성 시점 시장가 (limit_price, filled_price와 구분)
- **4전략 + 현금 10%**: crypto(10%), etf_mr(20%), dual_mom(15%), treasury_park(45% BIL)
- **market_hedge NO-GO**: 코드 유지하되 allocation=0 (SPY+SH 충돌, 변동성 감쇠)
- **교차 전략 심볼 합산 노출 관리**: max_symbol_exposure=20% (BTC 과집중 방지)
- **SHORT 시그널 지원**: order_builder에서 Alpaca SHORT 주문 생성 가능

## Testing
- Walk-forward tests: IS_RATIO=0.30, OOS_RATIO=0.10 (기본 60/20은 데이터 부족으로 윈도우 수 확보 불가)
- numpy bool 비교: `is` 대신 `==` 또는 `bool()` 사용 (`np.True_ is True` → False)
- External API providers (upbit, yfinance) 및 거래소 executor (upbit_executor, alpaca_executor) 커버리지 제외 — integration test scope
- Profit factor: MAX_PROFIT_FACTOR=100.0 캡 적용 (inf가 Go/No-Go 통과 방지)
- `backtest_weights()`와 `generate_signals()`는 반드시 동일한 수학 함수 사용 (EP1 data leakage, EP2 z-score 불일치 — 2회 CRITICAL)
- `generate_signals()`에 리밸런싱 게이트 필수: crypto=holding_days, dual=monthly rebalance_day, market_hedge=holding_days (EP3 daily rebalancing 재발 방지)
- ETF Mean Reversion은 z-score 임계치 기반이므로 리밸런싱 게이트 불필요 (매일 체크가 정상)
- Pairs Trading: `_compute_spread_zscore()`가 generate_signals + backtest_weights 공유 (동일 수학 함수 원칙)
- Backtest engine은 음수 weight 지원 (short leg = negative weight, long leg = positive weight)

## Conventions
- 한국어 설명, 영어 코드/변수명
- Phase 단위 커밋 (feat: implement Phase N — ...)
- Config: YAML + Pydantic 검증, 비밀값은 `${ENV_VAR}` 환경변수 참조
- 거래소: Upbit (crypto KRW), Alpaca (US ETF + inverse ETF)

## Paper Trading
- CLI: `uv run python -m hedgefund paper-run [--dry-run] [--verbose]`, `paper-reset [--yes]`
- 구동 주기: 일 1회 (매일 21:00 KST 권장)
- 리밸런싱 게이트: crypto=14일(격주), dual_momentum=월간, etf=매일(z-score)
- **주말 필터링**: US 시장 휴장일(토/일)에 Alpaca 전략 스킵, crypto+dual_momentum만 실행
- 상태 저장: `data/paper_state/{upbit,alpaca,strategies}.json`
- Two PaperExecutors: UPBIT (KRW) + ALPACA (USD), 자본 분할 기반
- **전략별 포지션 격리**: `PaperPosition`에 `strategy_name` 포함, `_positions` 키 = `(strategy_name, symbol)` 튜플. SELL 시 동일 전략 포지션만 청산. `get_account_info().positions`는 심볼별 aggregate (Protocol 호환)
- `get_strategy_position_quantity(strategy_name, symbol)`: 전략별 포지션 조회 (PaperExecutor 전용)
- Executor state version: v2 (positions를 list로 직렬화, strategy_name 포함). v1 파일은 자동 거부 → fresh start

## Strategies
| Strategy | Allocation | Exchange | Direction | 역할 |
|----------|-----------|----------|-----------|------|
| crypto_momentum | 10% | Upbit | Long-only | 크립토 모멘텀 |
| etf_mean_reversion | 20% | Alpaca | Long-only | ETF 평균회귀 |
| dual_momentum | 15% | Both | Long + 방어 바스켓 | 교차자산 모멘텀 |
| market_hedge | 0% (NO-GO) | Alpaca | 비활성 | SPY+SH 충돌 |
| treasury_park | 45% | Alpaca | BIL 상시 홀딩 | 유휴 현금 → 무위험 수익 |
| pairs_trading | 0% (조건부) | Alpaca | Long-short 쌍 | market-neutral alpha (WF 검증 후 활성화) |
| sector_momentum | 0% (조건부) | Alpaca | Long-only 섹터 ETF | optimize로 검증 후 활성화 |
| 현금 버퍼 | 10% | - | - | 긴급 리밸런싱 여유 |

## Risk Limits
- Max DD: 20%, 단일 포지션: 15%, 전략 배분: 40%, 심볼 합산: 20%
- VaR 95%: 2%, 스톱로스: 5% (코인 변동성 고려), 레버리지: 1.0x

## Key Paths
- Config: `config/settings.yaml`, `config/strategies/*.yaml`
- Core: `src/hedgefund/core/` (models, enums, exceptions, risk_metrics, kelly)
- Strategies: `src/hedgefund/strategies/` (base, crypto_momentum, etf_mean_reversion, dual_momentum, market_hedge, treasury_park, sector_momentum, pairs_trading, registry)
- Backtest: `src/hedgefund/backtest/` (engine, walk_forward, metrics, deflated_sharpe)
- Risk: `src/hedgefund/risk/` (manager, limits, drawdown, position_sizer)
- Portfolio: `src/hedgefund/portfolio/` (manager, allocator, correlation, rebalancer)
- Execution: `src/hedgefund/execution/` (protocols, cost_model, order_builder, state, executors/)
- App: `src/hedgefund/app.py` (wiring), `cli.py` (typer), `__main__.py`
- Scheduler: `src/hedgefund/scheduler/runner.py`

## Harness (Autonomous Feedback Loop)
- **Hooks**: `.claude/hooks/` — auto-format(ruff), file-protection(.env/paper_state), stop-test-verification, desktop-notification
- **Slash Commands**: `/validate`, `/paper-check`, `/entropy-check`, `/risk-audit`, `/strategy-review`
- **Architecture Tests**: `tests/test_architecture.py` — dependency layering, frozen dataclass, Strategy Protocol, file size, circular import, risk config, Executor Protocol (기계적 강제)
- **Trace Filesystem**: `.claude/traces/` — 진화 이력(evolution/), 실패 진단(failures/), 실험(experiments/) raw context 보존
- **변경 전략**: Additive first → Subtractive → Structural (한 번에 하나, 교란 변수 격리)
- **Protected Files**: .env*, data/paper_state/*, data/hedgefund.db (에이전트 수정 차단)
- **Warning Files**: config/settings.yaml, pyproject.toml (수정 허용하되 경고)

## Critical Bug History
- z-score 버그 (Phase 6 수정): `mean(window) × lookback ≡ sum(window)` → 분자 항상 0 → ETF MR 비활성. CLT 기반 z = Σr / (σ√N)으로 수정
- 교차 전략 노출 미관리 (Phase 6 추가): BTC가 crypto+dual에서 최대 41.7% → check_symbol_aggregate_exposure() 추가
- Market Hedge NO-GO (Phase 6 리뷰): SPY+SH 동시 보유 충돌, 인버스 ETF 변동성 감쇠 -4%/년, 단일 레짐 전용 → allocation 0으로 비활성화
- 전략 간 포지션 격리 부재 (Paper Trading 중간점검): PaperExecutor가 심볼 단위로만 포지션 관리 → ETF MR이 Dual Momentum의 TLT/GLD 포지션 강제 청산. `(strategy_name, symbol)` 복합 키로 격리 구현, state v2, 오염 데이터 리셋
- ETF MR z_entry=-1.5 구조적 미도달 (Paper Trading 중간점검): drift correction으로 SPY/QQQ z-score 최솟값 -1.08, -1.5 도달 불가. 7.8개월 OHLCV 분석 후 -1.0으로 완화 (80% 승률, 5일 +1.02%)

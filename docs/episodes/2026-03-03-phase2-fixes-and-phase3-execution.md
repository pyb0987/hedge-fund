# 2026-03-03: Phase 2 전략 버그 수정 + Phase 3 실행 레이어 구현

## Causal Chain

### 1. ETF Mean Reversion z-score 불일치 (CRITICAL)
- Symptom: Senior PM 리뷰에서 `generate_signals()`와 `backtest_weights()`가 다른 z-score 함수 사용 발견
- Investigation: `generate_signals()` → `_compute_rolling_zscore()` (누적 로그수익률 z-score), `backtest_weights()` → `BaseStrategy.compute_zscore()` (단일일 수익률 z-score)
- Root cause: 백테스트와 실제 시그널 생성의 수학적 로직이 불일치 → 백테스트 결과가 실거래를 대표하지 않음
- Resolution: `backtest_weights()`도 `_compute_rolling_zscore()`를 사용하도록 통일. 추가로 `_compute_rolling_zscore()` 내부 수학을 로그수익률 기반으로 개선

### 2. Crypto Momentum 일일 리밸런싱 (CRITICAL)
- Symptom: `holding_days=7` 설정값이 있지만 `backtest_weights()`에서 매일 가중치를 재계산
- Investigation: 코드에서 `holding_days`를 읽지만 실제로는 모든 날짜에 모멘텀 계산 수행
- Root cause: `for i, date in enumerate(dates)` 루프에서 리밸런스 간격 체크 없이 매번 재계산
- Impact: 소액 펀드에서 ~15% 연간 비용 드래그 추정 (주간→일간 회전율 7배)
- Resolution: carry-forward 로직 추가 — `days_since_rebalance >= holding_days`일 때만 새 가중치 계산, 그 외에는 이전 가중치 유지

### 3. Dual Momentum 자산 하드코딩
- Symptom: YAML config에 `assets` 섹션이 있지만 코드는 모듈 상수 `OFFENSIVE_ASSETS`, `DEFENSIVE_ASSET` 사용
- Resolution: 모듈 상수 제거, config에서 `offensive_assets`/`defensive_asset` 읽도록 변경, YAML 로더에 `assets` 섹션 파싱 추가

### 4. Order 모델 `price` 필드 누락
- Symptom: Phase 3 테스트 실행 시 `test_order_builder.py`가 통과했지만, `order_builder.py`에서 `Order(price=current_price)`로 생성 — Order에 `price` 필드 없음
- Investigation: `grep order.price` → 4곳에서 사용 (order_builder, portfolio manager, upbit executor, alpaca executor)
- Root cause: Phase 3 구현 시 Order 모델에 `price` 필드 추가를 빠뜨림
- Resolution: `Order`에 `price: float | None = None` 필드 추가

### 5. Paper Executor PnL 계산 순서 버그
- Symptom: `test_sell_records_pnl` 실패 — 매도 TradeRecord의 pnl이 None
- Hypothesis: sell 후 entry_price를 가져오는데 position이 이미 삭제됨
- Investigation: `_update_position_sell()` 호출 후 `self._positions.get(order.symbol)` → None (전량 매도 시 position 삭제됨)
- Root cause: entry_price 캡처를 position 삭제 후에 시도
- Resolution: `sell_entry_price = current_pos.avg_entry_price`를 `_update_position_sell()` 호출 전에 저장

### 6. Test Fixture 불일치로 인한 Drawdown 오탐
- Symptom: `test_cycle_with_long_signal` 실패 — "Max drawdown breached: 50.0%"
- Investigation: RiskManager `initial_capital=1_000_000`, PaperExecutor `initial_cash=500_000` → portfolio value 500K vs peak 1M = 50% DD
- Root cause: fixture 간 initial capital 불일치
- Resolution: RiskManager `initial_capital=500_000`으로 수정

## Negative Knowledge

- **Order 생성자에 존재하지 않는 필드를 전달해도 즉시 에러가 안 남**: Python dataclass는 `__init__`에서 정의되지 않은 kwarg를 받으면 TypeError를 발생시키지만, 해당 함수를 직접 호출하는 테스트가 없으면 발견이 늦어짐. 반드시 생성 함수에 대한 통합 테스트 필요.
- **frozen dataclass에서 삭제 순서가 중요**: `_update_position_sell()`이 position을 dict에서 제거하므로, 제거 전에 필요한 데이터(entry_price)를 캡처해야 함. mutable state + frozen 모델 조합 시 순서 의존성 주의.
- **Test fixture 간 의미적 연결**: RiskManager의 `initial_capital`과 Executor의 `initial_cash`는 논리적으로 같은 값이어야 하지만 별도 fixture로 분리하면 불일치가 발생하기 쉬움.
- **Upbit/Alpaca executor는 단위 테스트 불가**: 외부 API에 완전히 의존하므로 mock 없이는 테스트 불가. integration test scope로 분류하고 coverage 계산에서 제외하는 것이 현실적.

## Cross References
- Related: docs/episodes/2026-03-02-initial-build-and-critical-bugs.md (Phase 1+2 초기 구축)
- Related: MEMORY.md (Phase 상태 추적)

## Implicit Preferences
- Phase 단위 큰 배치로 구현 후 한번에 테스트 작성하는 스타일 유지
- Senior PM 리뷰를 Phase 완료 시점에 실행하여 교차 검증
- 외부 API 의존 코드는 coverage에서 제외하는 실용적 접근

## Derived Rules
→ Suggest adding to CLAUDE.md: "Phase 3 Execution layer: Upbit/Alpaca executor는 외부 API 의존 → integration test scope"
→ Suggest adding to CLAUDE.md: "Order 모델에 `price` 필드 포함 (market price at creation, distinct from limit_price/filled_price)"
→ Suggest adding to MEMORY.md: Phase 3 상태 업데이트

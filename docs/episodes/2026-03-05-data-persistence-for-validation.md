# 2026-03-05: Paper Trading 데이터 영속화 확장

## Topic 1: 검증 불가능 데이터 갭 식별

### Process
- 사용자 질문: "한 달간 paper run에서 검증할 것은 무엇이고, 데이터가 잘 쌓이고 있는가?"
- 기존 DB 스키마(signals, trades, portfolio_snapshots, risk_events) vs 검증 필요 항목 대조 분석
- Agent로 심층 감사 수행 → 6가지 갭 식별

### 발견된 6가지 갭
| 항목 | 심각도 | 이유 |
|------|--------|------|
| 전략별 성과 분리 | CRITICAL | snapshot에 전략별 기여도 없음 → 어떤 전략이 손실 원인인지 모름 |
| 포지션 보유 현황 | CRITICAL | executor 메모리에만 존재, DB에 안 쌓임 |
| 리밸런싱 게이트 감사 | HIGH | should_rebalance()=False일 때 아무 기록 없음 |
| 전략별 목표/실제 배분율 | HIGH | DD multiplier, target allocation 미저장 |
| 보유 기간 분석 | HIGH | 매수↔매도 연결 없음 |
| OHLCV 시장 데이터 | MEDIUM | 수집 후 버려짐 |

### Negative Knowledge
- signals+trades만으로는 전략별 성과를 분리할 수 없음: trades에 strategy_name이 있어도 "현재 보유 포지션"은 알 수 없음
- portfolio_snapshots의 total_value/cash만으로는 포지션 레벨 분석 불가
- OHLCV save_ohlcv()는 이미 구현되어 있었지만 app.py에서 호출하지 않고 있었음

## Topic 2: Phantom Drawdown 버그 + 미전파 리팩터링

### Process
- 이전 세션에서 `_create_strategies(config_dir, settings)`로 시그니처 변경했으나 caller `run_once()`에서 `settings` 미전달 → TypeError
- deployed_capital 수정도 이전 세션 발견 → 이번 세션에서 커밋
- 두 가지를 한 커밋에 묶어 처리

### Negative Knowledge
- RiskManager에 initial_capital=13M 전달하면 executors 배포 자본(11.7M)과 불일치 → day1에 10% phantom drawdown 발생
- 시그니처 변경 후 caller를 반드시 같이 수정해야 함 (Python은 런타임까지 감지 못함)

## Topic 3: 코드 리뷰 수정

### Process
- security-reviewer + python-reviewer 병렬 실행
- CRITICAL 2건: private `_positions`/`_trades` 접근 (캡슐화 위반), 미임포트 타입의 문자열 forward reference
- HIGH: crypto/sector momentum의 get_rebalance_decision() 중복 → BaseStrategy._holding_day_rebalance_decision() 추출

### Negative Knowledge
- `executor._trades` 직접 접근은 public `executor.trades` 프로퍼티가 이미 존재함에도 사용됨 → 항상 public API 우선
- Pydantic v2에서 `@field_validator`는 default value에 대해 실행되지 않음 → `@model_validator(mode="after")` 사용 필요

## Cross References
- Related: docs/episodes/2026-03-03-phase2-fixes-and-phase3-execution.md (Phase 4 리밸런싱 게이트)
- Related: CLAUDE.md — Paper Trading 섹션

## Derived Rules
→ CLAUDE.md에 추가 제안: Paper Trading 검증 데이터 관련 DB 테이블 목록
→ CLAUDE.md에 추가 제안: PaperExecutor public API 사용 규칙 (positions, trades 프로퍼티)

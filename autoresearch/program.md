# Auto-Search Program — Hedge Fund

## Identity

You are an autonomous portfolio researcher for a quant hedge fund.
Operate autonomously. Do NOT pause for confirmation.
Stop at 100 experiments or hypothesis exhaustion.

## Objective

Maximize **Information Ratio** (alpha / tracking_error).
- alpha = Jensen's alpha (portfolio return unexplained by market beta)
- tracking_error = residual volatility after removing SPY beta
- Current best: see `autoresearch/baseline.json`

**핵심 목표**: 시장수익률(SPY)과 무관한 절대수익 생성. 이것이 hedge fund의 존재 이유다.

## What You Can Modify (Mutable Genome)

### Strategy Logic (.py)
- `src/hedgefund/strategies/crypto_momentum.py` — 시그널 로직, 모멘텀 계산, 필터
- `src/hedgefund/strategies/etf_mean_reversion.py` — z-score 계산, 진입/청산 로직, 필터
- `src/hedgefund/strategies/dual_momentum.py` — 교차자산 모멘텀, 방어 전환 로직
- `src/hedgefund/strategies/sector_momentum.py` — 섹터 선별 로직, 모멘텀 계산
- `src/hedgefund/strategies/treasury_park.py` — (제한적) 파킹 로직

### Strategy Config (.yaml)
- `config/strategies/crypto_momentum.yaml` — lookback, holding_days, top_n
- `config/strategies/etf_mean_reversion.yaml` — lookback, z_entry, z_exit, drift_correction
- `config/strategies/dual_momentum.yaml` — lookback, defensive weights
- `config/strategies/sector_momentum.yaml` — lookback, holding_days, top_n
- `config/strategies/treasury_park.yaml` — symbol

### Portfolio Allocation
- `config/settings.yaml` — `allocation:` 섹션만 수정 가능 (risk 섹션 수정 금지)

## What You CANNOT Modify (Immutable)

- `autoresearch/evaluate.py` — 평가기 (수정 시 자기 평가 오염)
- `autoresearch/program.md` — 이 파일
- `src/hedgefund/backtest/` — 백테스트 엔진 (평가 인프라)
- `src/hedgefund/core/` — 도메인 모델, risk_metrics, enums, exceptions
- `src/hedgefund/risk/` — 리스크 관리
- `src/hedgefund/strategies/base.py` — Strategy Protocol/ABC
- `src/hedgefund/strategies/registry.py` — 레지스트리
- `src/hedgefund/config/schemas.py` — Pydantic 스키마
- `src/hedgefund/config/loader.py` — 설정 로더
- `tests/` — 테스트 스위트
- `data/` — 저장 데이터

## Experiment Loop

```
1. Read program.md (방향 + rejection history)        [첫 세션만]
   → "What You Can Modify" 섹션에서 가변 유전체 파일 경로를 확인
2. Read experiments.jsonl → last n → resume from n+1 [매 세션]
3. Formulate hypothesis (1-line, rejection history 중복 불가)
4. Implement change (program.md에 명시된 가변 유전체 파일만 수정)
5. git commit -m "experiment: [hypothesis]"
6. Run: uv run python autoresearch/evaluate.py → parse JSON
7. ADOPT → keep + baseline 갱신 + log
   REJECT → git reset --hard HEAD~1 + log
8. Repeat from 3 (budget 또는 consecutive reject limit까지)
```

## Adoption Criteria

- Information Ratio improvement ≥ **5%** over current best
- Guards must ALL PASS:
  - `tests`: 전체 테스트 통과
  - `beta`: |β| ≤ 0.30 (시장 독립성)
  - `max_dd`: OOS MaxDD ≤ 20%
  - `efficiency`: Walk-forward efficiency > 0.50 (과적합 방지)
  - `sharpe_positive`: OOS Sharpe > 0

## Structural Constraints (NEVER violate)

1. **Frozen dataclasses**: 모든 도메인 모델은 immutable
2. **Strategy Protocol + ABC**: `base.py`의 Protocol/ABC 패턴 준수
3. **Vectorized backtest**: weight matrix 방식. event-loop 금지
4. **backtest_weights() ↔ generate_signals() 동일 수학**: 두 함수는 동일한 계산을 사용해야 함 (2회 CRITICAL 버그 이력)
5. **Rebalancing gates**: crypto=holding_days, dual=monthly rebalance_day, sector=holding_days. ETF MR은 z-score 기반이므로 게이트 불필요
6. **Quarter-Kelly**: full Kelly 금지 (생존 우선)
7. **Max symbol exposure 20%**: 교차 전략 심볼 합산 한도
8. **Risk limits 수정 금지**: `config/settings.yaml`의 risk 섹션
9. **Allocation 합계 ≤ 1.0**: cash buffer 포함
10. **Market Hedge NO-GO**: allocation 0 유지 (SPY+SH 충돌)

## Research Axes & Strategy

### 1차 축: 시장 베타 감소 (beta → 0)
현재 포트폴리오의 SPY 베타를 측정하고, 상관관계를 줄이는 방향으로 탐색.
- 전략 간 상관관계 분석 → 역상관 강화
- Dual momentum의 방어 전환 조건 최적화
- Treasury park 비중 조정 (무위험 = beta 0)

### 2차 축: 알파 생성 (alpha 증대)
시장 독립적 초과수익 원천 발굴.
- 전략별 신호 품질 개선 (노이즈 필터링)
- 새로운 팩터/지표 추가 (기존 전략 내)
- 리밸런싱 타이밍 최적화

### 3차 축: 전략 간 시너지
포트폴리오 수준에서 전략 조합 효과 극대화.
- Allocation weight 최적화
- 교차 전략 상관관계 기반 동적 배분
- 스트레스 시나리오에서 방어력 강화

## Rejection History — EXHAUSTED AXES (DO NOT REVISIT)

(아직 없음 — 실험 진행에 따라 인간이 업데이트)

## Promising Unexplored Directions (Hints)

- VIX 기반 레짐 필터 (고변동성 시 현금 비중 증가)
- 이동평균 크로스오버를 모멘텀 확인 필터로 사용
- 달러 인덱스(DXY) 기반 crypto-ETF 배분 스위칭
- 볼린저 밴드 %B를 z-score 보조 지표로 결합
- Funding rate (crypto) 기반 과열 감지
- 섹터 모멘텀 활성화 (현재 allocation 0) + 적절한 비중 배분

## Session Handoff Protocol

세션 종료 시 `.claude/handoff.md` 작성:
```
## Status: in_progress | blocked
## Last experiment: n=<N>
## Current hypothesis: <if mid-experiment>
## Git state: clean | dirty
## Next: <suggested next hypothesis>
```

다음 세션: handoff.md + experiments.jsonl 마지막 항목에서 재개.

## Logging Format (experiments.jsonl)

```json
{"ts": "ISO8601", "n": 1, "hypothesis": "...", "ir": 0.42, "alpha": 0.03, "beta": 0.15, "improvement_pct": 5.2, "verdict": "ADOPT", "guards": {"tests": "PASS", "beta": "PASS", ...}, "sha": "abc123", "reverted": false}
```

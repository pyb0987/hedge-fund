# 2026-03-02: Phase 1+2 초기 구축 및 치명적 버그 3건 발견

## Causal Chain

### 1. Walk-Forward OOS 데이터 누수
- Symptom: Senior PM 리뷰 중 walk_forward.py:103에서 OOS 가중치 생성 코드 발견
- Hypothesis: OOS 가중치를 생성할 때 IS 데이터만 사용해야 하는데 IS+OOS 전체를 전달하고 있음
- Investigation: `prices.iloc[is_start_idx:oos_end_idx]`는 IS 시작~OOS 끝까지 포함 → 미래 가격이 파라미터 피팅에 포함됨
- Root cause: 주석은 "Use IS data for parameter fitting"이라고 했지만 코드는 IS+OOS 데이터를 전달
- Resolution: `is_prices[[col]]`만 전달하도록 수정 — OOS 데이터 완전 격리

### 2. Profit Factor inf
- Symptom: `profit_factor()`가 손실 0일 때 `float("inf")` 반환
- Hypothesis: Go/No-Go 검증에서 `inf >= 1.3`이 항상 True → 의미없는 데이터에서도 통과
- Root cause: 손실이 없는 구간(예: 백테스트 초기 평탄 구간)에서 inf 반환
- Resolution: MAX_PROFIT_FACTOR=100.0으로 캡. 실제 전략에서 PF>100은 비현실적

### 3. Python 3.9 환경 문제
- Symptom: `pip install -e ".[dev]"` 실패 — "editable mode requires setuptools-based build"
- Hypothesis: 시스템 Python이 3.9.6으로 hatchling editable install 미지원
- Investigation: `python3 --version` → 3.9.6, pyproject.toml은 `requires-python = ">=3.11"`
- Resolution: `uv venv --python 3.12`로 새 venv 생성 후 `uv pip install`

### 4. Walk-Forward 테스트 윈도우 부족
- Symptom: 500일 데이터로 walk-forward 테스트 시 "Only 1 windows generated" 에러
- Hypothesis: IS 60% + OOS 20% = 80% → 500 * 0.8 = 400일 윈도우, step=100 → 2개만 가능
- Investigation: min_windows=3인데 데이터가 부족함
- Resolution 1차: 1500일로 증가 → 여전히 2개만 생성 (1200일 윈도우, step=300)
- Root cause: 기본 비율(IS 60%+OOS 20%)이 너무 크면 매우 많은 데이터 필요
- Resolution 최종: 테스트에서 IS_RATIO=0.30, OOS_RATIO=0.10으로 축소

## Negative Knowledge

- **Walk-Forward 기본 비율 (60/20)은 1000일 이상 데이터 필요**: IS 60%+OOS 20% 비율에서 window_size = total * 0.80, step = total * 0.20. 3개 윈도우: total * 0.80 + 2 * total * 0.20 ≤ total → 1.2 * total ≤ total (불가능). 결국 step이 oos_size와 같으면 (windows - 1) * oos_size + window_size ≤ total 이므로 최소 total = window_size + 2 * step = 0.8N + 2 * 0.2N = 1.2N → 불가능. 실제로는 정수 변환 때문에 겨우 2개가 나옴.
- **`pip install -e` with hatchling**: 구형 pip (21.2.4)에서는 hatchling editable install 미지원. `uv`를 사용하면 문제없음.
- **numpy bool vs Python bool**: `np.True_ is True`는 False. 테스트에서 `is` 대신 `bool()` 래핑 또는 `==` 사용 필요.
- **Frozen dataclass의 dict 필드**: `metadata: dict | None`은 frozen이어도 dict 내부는 mutable. 순수 불변성이 필요하면 `MappingProxyType` 사용 고려.

## Cross References
- Related: MEMORY.md (프로젝트 상태 추적)
- Hedge Fund Senior Manager skill 활용하여 코드 리뷰 수행

## Implicit Preferences
- 사용자는 한국어로 설명을 선호하되 코드/변수명은 영어
- 태스크 리스트를 통한 진행 상황 추적을 선호
- TDD보다 구현-후-테스트 방식으로 진행 (Phase 1 전체 구현 → 테스트 한번에 작성)
- 커밋은 Phase 단위로 큰 단위 선호

## Derived Rules
→ Suggest adding to project CLAUDE.md: "Walk-forward 테스트 시 IS/OOS 비율을 0.30/0.10으로 줄여서 충분한 윈도우 수 확보"
→ Suggest adding to project CLAUDE.md: "numpy bool 비교 시 `is` 대신 `==` 또는 `bool()` 사용"

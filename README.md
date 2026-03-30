# Hedge Fund — 소액 퀀트 자동매매 시스템

$10,000 / 1,300만원으로 시작하는 자동화된 퀀트 헤지펀드 시스템.
**목표는 수익 극대화가 아니라 시장에서의 생존(자본 보존 우선, 성장 차선)**.

4개의 비상관 전략 + 45% 무위험 주차(Treasury)를 동시 운용하여
단일 전략 실패 리스크를 분산하고, 엄격한 리스크 관리 규칙으로 자본을 보호합니다.

> **Paper Trading 진행 중** — 3개월 검증(Go/No-Go) 후 실거래 전환 예정

---

## 시스템 개요

```
[Upbit API]     [Alpaca API]     [yfinance]
     │               │               │
     └───────┬───────┘───────┬───────┘
             │               │
         ┌───▼───┐     ┌────▼────┐
         │ Data  │     │  Data   │
         │Upbit  │     │yfinance │
         └───┬───┘     └────┬────┘
             │              │
     ┌───────▼──────────────▼────────┐
     │         Strategy Engine        │
     │  A: Crypto Momentum   (10%)   │
     │  B: ETF Mean Reversion (20%) │
     │  C: Dual Momentum     (15%)  │
     │  D: Treasury Park     (45%)  │
     │  Cash Buffer          (10%)  │
     └──────────────┬────────────────┘
                    │ Signals
              ┌─────▼──────┐
              │    Risk     │
              │  Manager    │
              └─────┬──────┘
                    │ Approved Orders
           ┌────────▼─────────┐
           │   Order Builder   │
           │ Signal → Order    │
           └────────┬─────────┘
                    │
        ┌───────────▼───────────┐
        │     Executors          │
        │  Paper │ Upbit │ Alpaca│
        └───────────┬───────────┘
                    │
           ┌────────▼────────┐
           │   Monitoring     │
           │ Report│Telegram  │
           └─────────────────┘
```

---

## 전략 구성

| 전략 | 배분 | 거래소 | 방향 | 역할 |
|------|:---:|--------|------|------|
| A: Crypto Momentum | 10% | Upbit | Long-only | 크립토 모멘텀 (격주 리밸런싱) |
| B: ETF Mean Reversion | 20% | Alpaca | Long-only | ETF z-score 평균회귀 |
| C: Dual Momentum | 15% | Both | Long + 방어 바스켓 | 교차자산 모멘텀 (월간) |
| D: Treasury Park | 45% | Alpaca | BIL 상시 홀딩 | 유휴 현금 → 무위험 수익 |
| Cash Buffer | 10% | - | - | 긴급 리밸런싱 여유 |

### 전략 A: 암호화폐 모멘텀

**원리**: 최근 가격이 많이 오른 코인이 단기적으로 계속 오르는 경향(모멘텀 프리미엄)을 이용.

- **유니버스**: KRW 거래량 상위 15개 코인 (동적 선정)
- **리밸런싱**: 14일(격주) — 거래 비용 최소화
- **매수**: 20일 모멘텀 양수인 코인 중 상위 3개에 균등 배분
- **매도**: 순위 밖으로 밀려나거나 모멘텀 음수 시 전량 매도

### 전략 B: 미국 ETF 평균회귀

**원리**: ETF 가격이 평균에서 크게 벗어나면 다시 돌아오는 경향(평균회귀)을 이용.

- **유니버스**: SPY, QQQ, TLT, GLD, IEF
- **리밸런싱**: 매일 체크 (z-score 임계치 기반)
- **매수**: z-score ≤ -1.0 (drift correction 적용)
- **매도**: z-score ≥ +1.5

### 전략 C: 듀얼 모멘텀

**원리**: Gary Antonacci의 듀얼 모멘텀 — 절대 모멘텀 + 상대 모멘텀 결합.

- **공격 자산**: BTC/KRW, SPY
- **방어 자산**: TLT, GLD (분산 바스켓)
- **리밸런싱**: 월간 (매월 1일)
- 둘 다 양수 → 승자에 100%, 둘 다 음수 → 방어 바스켓으로 전환

### 전략 D: Treasury Park

- **BIL** (1-3개월 미국 단기 국채 ETF) 상시 보유
- 연 ~5% 무위험 수익으로 유휴 자본 활용
- 리밸런싱 불필요 (보유 유지)

---

## 리스크 관리

모든 주문은 Risk Manager를 통과해야 실행됩니다.

### 하드 리밋

| 규칙 | 한도 | 위반 시 |
|------|:---:|---------|
| 최대 포트폴리오 드로다운 | 20% | 전 포지션 매매 중단 |
| 최대 단일 포지션 비중 | 15% | 해당 주문 거부 |
| 최대 단일 전략 비중 | 40% | 해당 주문 거부 |
| 교차 전략 심볼 합산 | 20% | BTC 과집중 방지 |
| 일일 VaR (95%) | 2% | 신규 매수 중단 |
| 스톱로스 | 5% | 기계적 매도 |
| 레버리지 | 1.0x | 현물만 |

### 드로다운 기반 점진적 포지션 감소

```
고점 대비 손실    포지션 규모
─────────────    ──────────
  0% ~  5%       100%  (정상 운영)
  5% ~ 10%        75%  (조기 경고)
 10% ~ 15%        50%  (절반 감소)
 15% ~ 20%        25%  (최소 포지션)
    > 20%           0%  (전체 매매 중단)
```

### 포지션 사이징

- **Quarter-Kelly**: Full Kelly의 25% — 성장률 75% 희생, 최대 손실 75% 감소
- **변동성 스케일링**: 목표 연변동성 15% 초과 시 자동 축소

---

## 거래 비용 모델

| 거래소 | 수수료 (편도) | 슬리피지 | 왕복 비용 | 최소 주문 |
|:---:|:---:|:---:|:---:|:---:|
| Upbit | 0.05% | 0.15% | 0.40% | 5,000 KRW |
| Alpaca | 0.00% | 0.05% | 0.10% | $1 |

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | Python 3.12 |
| 패키지 관리 | uv + hatchling |
| 거래소 SDK | pyupbit (Upbit), alpaca-trade-api (Alpaca) |
| 데이터 | yfinance (ETF), pyupbit (암호화폐) |
| 검증 | pydantic (설정), pytest (464 tests, 89% coverage) |
| 저장소 | SQLite |
| 스케줄링 | launchd (macOS) |
| 로깅 | structlog |
| 알림 | Telegram Bot API (httpx) |
| CLI | typer |

---

## Setup

```bash
# 1. Clone & install
git clone https://github.com/pyb0987/hedge-fund.git
cd hedge-fund
uv sync --all-extras

# 2. Environment variables
cp .env.example .env
# Edit .env with your API keys (Upbit, Alpaca, Telegram)

# 3. Config
cp config/settings.yaml.example config/settings.yaml
# Edit config/settings.yaml with your parameters

# 4. Run tests
uv run pytest tests/ -q

# 5. Paper trading (dry run)
uv run python -m hedgefund paper-run --dry-run

# 6. Install daily scheduler (macOS)
uv run python -m hedgefund install-scheduler
```

### CLI Commands

```bash
uv run python -m hedgefund paper-run [--dry-run] [--verbose]  # 1회 실행
uv run python -m hedgefund paper-status                        # 포트폴리오 상태
uv run python -m hedgefund paper-report                        # Go/No-Go 리포트
uv run python -m hedgefund validate-wf [--days N]              # Walk-Forward 검증
uv run python -m hedgefund optimize <strategy>                 # 파라미터 최적화
uv run python -m hedgefund daily-run [--force]                 # 일일 스케줄 실행
uv run python -m hedgefund install-scheduler                   # launchd 설치
```

---

## 프로젝트 구조

```
hedge-fund/
├── config/                    # YAML 설정 + 전략별 파라미터
├── src/hedgefund/
│   ├── core/                  # 도메인 모델 (frozen dataclass), 리스크 메트릭
│   ├── config/                # YAML → Pydantic 검증
│   ├── data/                  # 시장 데이터 수집 (Upbit, yfinance) + SQLite
│   ├── strategies/            # 매매 전략 (Protocol + ABC 패턴)
│   ├── risk/                  # 리스크 관리 (drawdown, limits, position_sizer)
│   ├── execution/             # 주문 실행 (Paper / Upbit / Alpaca executors)
│   ├── portfolio/             # 포트폴리오 관리 + 리밸런싱
│   ├── backtest/              # 벡터화 백테스트 + Walk-Forward + Deflated Sharpe
│   ├── monitoring/            # Paper trading 리포트 + Telegram 알림
│   └── scheduler/             # launchd 기반 일일 스케줄러
└── tests/                     # 464 tests (unit + architecture)
```

---

## 구현 진행 상황

| Phase | 상태 | 내용 |
|:---:|:---:|------|
| 1-5b | COMPLETE | 핵심 모델, 전략 A/B/C, 리스크, 포트폴리오, 실행, 백테스트, CLI |
| 6 | COMPLETE | 전략 리뷰, 리스크 구조조정, 교차 전략 노출 관리, 페이퍼 트레이딩 |
| 7 | NOT STARTED | Docker → Oracle Cloud 배포 |

---

## 핵심 설계 원칙

1. **생존 우선**: 15% CAGR + 10% MaxDD > 40% CAGR + 60% MaxDD
2. **과적합 경계**: Walk-Forward + Deflated Sharpe + 파라미터 안정성 검증
3. **비용 현실주의**: 소액에서 거래 비용이 알파를 잡아먹으므로 리밸런싱 빈도 최소화
4. **무레버리지**: 1.0x 고정 — 소액 계정은 마진콜을 버틸 수 없음
5. **분산 투자**: 비상관 전략 동시 운용 + 교차 전략 심볼 노출 관리

---

## License

Private — All rights reserved.

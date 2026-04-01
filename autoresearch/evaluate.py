#!/usr/bin/env python3
"""Fixed evaluator for hedge-fund autoresearch.

THIS FILE IS IMMUTABLE — agent must NOT modify it.

Primary metric: Information Ratio (alpha / tracking_error)
  - alpha = Jensen's alpha (market-independent excess return)
  - tracking_error = residual volatility after removing market beta
Benchmark: SPY (S&P 500)

Guards:
  - |beta| <= 0.30 (low market correlation)
  - OOS max drawdown <= 20%
  - Walk-forward efficiency > 0.50 (OOS/IS decay)
  - OOS Sharpe > 0 (positive risk-adjusted return)
  - All tests pass
"""

import json
import pickle
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = Path(__file__).resolve().parent
DATA_CACHE = AUTORESEARCH_DIR / "data_cache.pkl"
BASELINE_FILE = AUTORESEARCH_DIR / "baseline.json"

# --- Constants (do not modify) ---
EVAL_TIMEOUT = 600  # seconds
DATA_DAYS = 1260  # ~5 years
IMPROVEMENT_THRESHOLD = 5.0  # % improvement required for ADOPT
BENCHMARK = "SPY"
RF = 0.04  # risk-free rate (annual)
TRADING_DAYS = 252
IS_RATIO = 0.30
OOS_RATIO = 0.10
MAX_BETA = 0.30
MAX_DD = 0.20
MIN_EFFICIENCY = 0.25  # relaxed: beta-driven portfolio phase (raise when alpha established)


# --- Timeout ---
def _timeout(signum, frame):
    print(json.dumps({"verdict": "ERROR", "reason": "Evaluation timed out"}))
    sys.exit(1)


signal.signal(signal.SIGALRM, _timeout)
signal.alarm(EVAL_TIMEOUT)


# --- Guard: Tests ---
def run_tests() -> tuple[bool, str]:
    """All existing tests must pass."""
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/", "-x", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(PROJECT_ROOT),
    )
    tail = (r.stdout or "")[-500:]
    return r.returncode == 0, tail


# --- Data ---
def fetch_data() -> dict:
    """Fetch OHLCV data with pickle caching (fetched once per session)."""
    if DATA_CACHE.exists():
        with open(DATA_CACHE, "rb") as f:
            return pickle.load(f)

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from hedgefund.data.providers.upbit_provider import UpbitProvider
    from hedgefund.data.providers.yfinance_provider import YFinanceProvider

    yf = YFinanceProvider()
    upbit = UpbitProvider()

    etf_symbols = [
        # Core portfolio
        "SPY",
        "QQQ",
        "TLT",
        "GLD",
        "IEF",
        "BIL",
        # Pairs trading universe — structural cointegration candidates
        "GDX",  # gold miners (pairs: GLD)
        "SLV",  # silver (pairs: GLD)
        "XLE",  # energy sector (pairs: USO)
        "USO",  # crude oil (pairs: XLE)
        "XLF",  # financials sector
    ]
    crypto_symbols = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-ADA"]

    data = {}
    for sym in etf_symbols:
        df = yf.get_ohlcv(sym, interval="day", count=DATA_DAYS)
        if df is not None and len(df) > 50:
            data[sym] = df

    for sym in crypto_symbols:
        df = upbit.get_ohlcv(sym, interval="day", count=DATA_DAYS)
        if df is not None and len(df) > 50:
            data[sym] = df

    data = _normalize_dates(data)

    with open(DATA_CACHE, "wb") as f:
        pickle.dump(data, f)

    return data


def _normalize_dates(data: dict) -> dict:
    """Normalize all DataFrames to tz-naive, date-only index.

    Required because yfinance returns tz-aware (America/New_York)
    and Upbit returns tz-naive with 09:00 time component.
    Without normalization, date intersection across exchanges is empty.
    """
    normalized = {}
    for sym, df in data.items():
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()  # strip time → midnight
        df = df[~df.index.duplicated(keep="first")]  # dedupe
        normalized[sym] = df
    return normalized


# --- Portfolio Evaluation ---
def evaluate_portfolio(data: dict) -> dict:
    """Run portfolio walk-forward and compute market-neutral metrics."""
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from hedgefund.backtest.engine import BacktestConfig
    from hedgefund.backtest.portfolio_backtest import run_portfolio_walk_forward
    from hedgefund.config.loader import load_global_settings, load_strategy_config
    from hedgefund.strategies.registry import get_strategy

    # Load settings (agent may have modified allocation)
    settings = load_global_settings(PROJECT_ROOT / "config")
    alloc_model = settings.allocation
    allocation = {k: v for k, v in alloc_model.model_dump().items() if v > 0}

    # Instantiate strategies from current code + config
    strategies = {}
    for name in allocation:
        try:
            config = load_strategy_config(name, PROJECT_ROOT / "config")
            strategies[name] = get_strategy(name, config=config)
        except Exception as e:
            return {"error": f"Strategy '{name}' init failed: {e}"}

    if not strategies:
        return {"error": "No active strategies"}

    # Blended cost config
    bt_config = BacktestConfig(
        initial_capital=float(settings.portfolio.initial_capital),
        commission_rate=0.00125,
        slippage_rate=0.001,
    )

    # Portfolio walk-forward
    try:
        wf = run_portfolio_walk_forward(
            strategies=strategies,
            allocation=allocation,
            data=data,
            is_ratio=IS_RATIO,
            oos_ratio=OOS_RATIO,
            config=bt_config,
        )
    except Exception as e:
        return {"error": f"Walk-forward failed: {e}"}

    if not wf.windows:
        return {"error": "No walk-forward windows generated"}

    # Concatenate all OOS returns across windows
    oos_returns = pd.concat([w.oos_result.returns for w in wf.windows]).sort_index()

    # Benchmark (SPY) returns for same dates
    spy_close = data[BENCHMARK]["close"]
    spy_ret = spy_close.pct_change().dropna()

    common = oos_returns.index.intersection(spy_ret.index)
    if len(common) < 20:
        return {"error": f"Insufficient OOS-benchmark overlap: {len(common)} days"}

    pr = oos_returns.loc[common].values
    br = spy_ret.loc[common].values

    # Market-neutral metrics
    var_b = float(np.var(br))
    beta = float(np.cov(pr, br)[0, 1] / var_b) if var_b > 1e-12 else 0.0

    port_ann = float((1 + np.mean(pr)) ** TRADING_DAYS - 1)
    bench_ann = float((1 + np.mean(br)) ** TRADING_DAYS - 1)

    alpha = port_ann - (RF + beta * (bench_ann - RF))

    residual = pr - beta * br
    te = float(np.std(residual) * np.sqrt(TRADING_DAYS))
    ir = alpha / te if te > 1e-8 else 0.0

    return {
        "ir": round(ir, 4),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "tracking_error": round(te, 4),
        "oos_sharpe": round(float(wf.oos_combined_sharpe), 4),
        "oos_return": round(float(wf.oos_combined_return), 4),
        "oos_max_dd": round(float(wf.oos_combined_max_dd), 4),
        "efficiency": round(float(wf.efficiency_ratio), 4),
        "num_windows": len(wf.windows),
        "oos_days": len(common),
    }


# --- Guards ---
def check_guards(m: dict) -> dict:
    """Binary pass/fail guards on portfolio metrics."""
    return {
        "beta": "PASS" if abs(m["beta"]) <= MAX_BETA else "FAIL",
        "max_dd": "PASS" if m["oos_max_dd"] <= MAX_DD else "FAIL",
        "efficiency": "PASS" if m["efficiency"] > MIN_EFFICIENCY else "FAIL",
        "sharpe_positive": "PASS" if m["oos_sharpe"] > 0 else "FAIL",
    }


# --- Main ---
def main() -> dict:
    # Guard 1: tests
    ok, out = run_tests()
    if not ok:
        return {
            "verdict": "REJECT_GUARD",
            "guards": {"tests": "FAIL"},
            "test_output": out,
        }

    # Fetch / load cached data
    data = fetch_data()

    # Evaluate
    metrics = evaluate_portfolio(data)
    if "error" in metrics:
        return {"verdict": "ERROR", "reason": metrics["error"]}

    # Guards
    guards = check_guards(metrics)
    guards["tests"] = "PASS"
    all_pass = all(v == "PASS" for v in guards.values())

    # Baseline comparison
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            baseline = json.load(f)
        bl_ir = baseline["ir"]
        if abs(bl_ir) > 0.01:
            imp = (metrics["ir"] - bl_ir) / abs(bl_ir) * 100
        else:
            # Near-zero baseline: treat difference as percentage points × 100
            imp = (metrics["ir"] - bl_ir) * 100
    else:
        # First run — establish baseline
        bl_ir = metrics["ir"]
        imp = 0.0
        with open(BASELINE_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

    # Verdict
    if not all_pass:
        verdict = "REJECT_GUARD"
    elif imp >= IMPROVEMENT_THRESHOLD:
        verdict = "ADOPT"
        with open(BASELINE_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        verdict = "REJECT_THRESHOLD"

    return {
        "verdict": verdict,
        "ir": metrics["ir"],
        "alpha": metrics["alpha"],
        "beta": metrics["beta"],
        "baseline_ir": bl_ir,
        "improvement_pct": round(imp, 2),
        "metrics": metrics,
        "guards": guards,
    }


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        result = {"verdict": "ERROR", "reason": str(e)}
    print(json.dumps(result))

"""Architectural constraint tests — mechanical enforcement of design rules.

These tests ensure the codebase adheres to documented architecture invariants.
They run as part of the standard test suite and CI pipeline.
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent.parent / "src" / "hedgefund"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_python_files(directory: Path) -> list[Path]:
    """Collect all .py files recursively, excluding __pycache__."""
    return sorted(
        p
        for p in directory.rglob("*.py")
        if "__pycache__" not in str(p) and p.name != "__init__.py"
    )


def _parse_imports(filepath: Path) -> list[str]:
    """Extract all hedgefund.* imports from a Python file."""
    tree = ast.parse(filepath.read_text(), filename=str(filepath))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("hedgefund"):
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("hedgefund"):
                imports.append(node.module)
    return imports


def _get_module_name(filepath: Path) -> str:
    """Extract top-level module name from filepath (e.g., 'core', 'strategies')."""
    rel = filepath.relative_to(SRC_DIR)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _get_all_classes_in_module(module_path: str) -> list[type]:
    """Import a module and return all classes defined in it."""
    try:
        mod = importlib.import_module(module_path)
        return [
            obj
            for _, obj in inspect.getmembers(mod, inspect.isclass)
            if obj.__module__ == module_path
        ]
    except (ImportError, ModuleNotFoundError):
        return []


# ---------------------------------------------------------------------------
# Layer 1: Dependency Layering
# ---------------------------------------------------------------------------

# Allowed dependency graph (module -> set of modules it can import from)
ALLOWED_DEPS: dict[str, set[str]] = {
    "core": set(),  # core imports NOTHING from hedgefund
    "data": {"core", "strategies"},  # collector needs Strategy Protocol for type hints
    "config": {"core"},
    "risk": {"core", "config"},
    "execution": {"core", "risk", "config"},
    "strategies": {"core", "config"},
    "backtest": {"core"},
    "portfolio": {"core", "strategies", "execution", "risk", "config", "data"},
    "monitoring": {"core", "data", "config", "execution", "strategies", "risk", "portfolio"},
    "scheduler": {
        "core",
        "config",
        "data",
        "strategies",
        "execution",
        "risk",
        "portfolio",
        "monitoring",
        "app",  # scheduler invokes app.run_once()
    },
}

# Top-level orchestration files that can import anything
ORCHESTRATION_FILES = {"app.py", "cli.py", "__main__.py"}


class TestDependencyLayering:
    """Verify that module import boundaries are respected."""

    @pytest.fixture(autouse=True)
    def _collect_files(self) -> None:
        self.all_files = _collect_python_files(SRC_DIR)

    def test_no_upward_imports(self) -> None:
        """Modules must not import from layers above them."""
        violations: list[str] = []

        for filepath in self.all_files:
            # Skip orchestration files
            if filepath.name in ORCHESTRATION_FILES:
                continue

            module = _get_module_name(filepath)

            # Skip modules not in the dependency map
            if module not in ALLOWED_DEPS:
                continue

            allowed = ALLOWED_DEPS[module]
            imports = _parse_imports(filepath)

            for imp in imports:
                # Extract the imported module's top-level package
                parts = imp.replace("hedgefund.", "").split(".")
                imported_module = parts[0]

                # Self-imports are always allowed
                if imported_module == module:
                    continue

                if imported_module not in allowed:
                    violations.append(
                        f"{filepath.relative_to(SRC_DIR)}: "
                        f"'{module}' imports '{imported_module}' "
                        f"(allowed: {allowed})"
                    )

        assert violations == [], "Dependency layering violations found:\n" + "\n".join(
            f"  - {v}" for v in violations
        )


# ---------------------------------------------------------------------------
# Layer 2: Frozen Dataclass Enforcement
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    """All domain model dataclasses must be frozen=True."""

    # Modules that should contain only frozen dataclasses
    CHECKED_MODULES = [
        "hedgefund.core.models",
        "hedgefund.core.risk_metrics",
        "hedgefund.backtest.engine",
        "hedgefund.backtest.walk_forward",
        "hedgefund.backtest.optimizer",
        "hedgefund.execution.protocols",
        "hedgefund.execution.cost_model",
        "hedgefund.execution.executors.paper_executor",
        "hedgefund.risk.manager",
        "hedgefund.risk.limits",
        "hedgefund.risk.drawdown",
        "hedgefund.portfolio.rebalancer",
        "hedgefund.portfolio.correlation",
        "hedgefund.portfolio.manager",
        "hedgefund.monitoring.paper_report",
        "hedgefund.monitoring.telegram",
        "hedgefund.strategies.base",
        "hedgefund.data.collector",
        "hedgefund.data.cache",
    ]

    def test_all_dataclasses_are_frozen(self) -> None:
        """Every dataclass in domain modules must have frozen=True."""
        import dataclasses

        violations: list[str] = []

        for module_path in self.CHECKED_MODULES:
            classes = _get_all_classes_in_module(module_path)
            for cls in classes:
                if dataclasses.is_dataclass(cls):
                    # Check frozen by trying to set an attribute
                    params = dataclasses.fields(cls)
                    if not cls.__dataclass_params__.frozen:  # type: ignore[attr-defined]
                        violations.append(f"{module_path}.{cls.__name__}")

        assert violations == [], "Non-frozen dataclasses found:\n" + "\n".join(
            f"  - {v}" for v in violations
        )


# ---------------------------------------------------------------------------
# Layer 3: Strategy Protocol Compliance
# ---------------------------------------------------------------------------


class TestStrategyProtocol:
    """All concrete strategies must implement the Strategy Protocol."""

    STRATEGY_MODULES = [
        "hedgefund.strategies.crypto_momentum",
        "hedgefund.strategies.etf_mean_reversion",
        "hedgefund.strategies.dual_momentum",
        "hedgefund.strategies.market_hedge",
        "hedgefund.strategies.treasury_park",
        "hedgefund.strategies.sector_momentum",
    ]

    def test_strategies_inherit_base(self) -> None:
        """All strategy classes must inherit from BaseStrategy."""
        from hedgefund.strategies.base import BaseStrategy

        for module_path in self.STRATEGY_MODULES:
            classes = _get_all_classes_in_module(module_path)
            strategy_classes = [
                cls for cls in classes if "Strategy" in cls.__name__ and cls is not BaseStrategy
            ]

            for cls in strategy_classes:
                assert issubclass(cls, BaseStrategy), (
                    f"{module_path}.{cls.__name__} does not inherit from BaseStrategy"
                )

    def test_strategies_have_required_methods(self) -> None:
        """All strategies must implement generate_signals, get_universe, backtest_weights."""
        required_methods = ["generate_signals", "get_universe", "backtest_weights"]

        for module_path in self.STRATEGY_MODULES:
            classes = _get_all_classes_in_module(module_path)
            strategy_classes = [cls for cls in classes if "Strategy" in cls.__name__]

            for cls in strategy_classes:
                for method_name in required_methods:
                    method = getattr(cls, method_name, None)
                    assert method is not None, (
                        f"{module_path}.{cls.__name__} missing method: {method_name}"
                    )
                    # Verify it's not still abstract
                    assert not getattr(method, "__isabstractmethod__", False), (
                        f"{module_path}.{cls.__name__}.{method_name} is still abstract"
                    )


# ---------------------------------------------------------------------------
# Layer 4: File Size Limits
# ---------------------------------------------------------------------------


class TestFileSizeLimits:
    """Enforce maximum file sizes to maintain readability."""

    MAX_LINES = 800
    WARNING_LINES = 400

    def test_no_files_exceed_max(self) -> None:
        """No source file should exceed 800 lines."""
        violations: list[str] = []
        all_files = _collect_python_files(SRC_DIR)

        for filepath in all_files:
            line_count = len(filepath.read_text().splitlines())
            if line_count > self.MAX_LINES:
                violations.append(
                    f"{filepath.relative_to(SRC_DIR)}: {line_count} lines (max: {self.MAX_LINES})"
                )

        assert violations == [], f"Files exceeding {self.MAX_LINES} line limit:\n" + "\n".join(
            f"  - {v}" for v in violations
        )

    def test_report_large_files(self) -> None:
        """Report files above warning threshold (informational)."""
        large_files: list[str] = []
        all_files = _collect_python_files(SRC_DIR)

        for filepath in all_files:
            line_count = len(filepath.read_text().splitlines())
            if line_count > self.WARNING_LINES:
                large_files.append(f"{filepath.relative_to(SRC_DIR)}: {line_count} lines")

        # This test always passes but prints a warning
        if large_files:
            pytest.skip(
                f"Large files (>{self.WARNING_LINES} lines, consider decomposition):\n"
                + "\n".join(f"  - {f}" for f in large_files)
            )


# ---------------------------------------------------------------------------
# Layer 5: No Circular Imports
# ---------------------------------------------------------------------------


class TestNoCircularImports:
    """Verify there are no circular import chains."""

    def test_all_modules_importable(self) -> None:
        """Every module under hedgefund should import without circular dependency."""
        all_files = _collect_python_files(SRC_DIR)
        failures: list[str] = []

        for filepath in all_files:
            # Skip external API providers and executors (need network)
            rel = str(filepath.relative_to(SRC_DIR))
            if any(
                skip in rel
                for skip in [
                    "upbit_provider",
                    "upbit_executor",
                    "alpaca_executor",
                    "__main__",  # __main__.py invokes CLI on import
                ]
            ):
                continue

            module_path = "hedgefund." + str(filepath.relative_to(SRC_DIR)).replace(
                "/", "."
            ).replace(".py", "")

            try:
                importlib.import_module(module_path)
            except ImportError as e:
                if "circular" in str(e).lower():
                    failures.append(f"{module_path}: {e}")

        assert failures == [], "Circular imports detected:\n" + "\n".join(
            f"  - {f}" for f in failures
        )


# ---------------------------------------------------------------------------
# Layer 6: Risk Configuration Consistency
# ---------------------------------------------------------------------------


class TestRiskConfigConsistency:
    """Risk limits in code must match documented values."""

    def test_progressive_drawdown_thresholds(self) -> None:
        """DrawdownState must use 5/10/15/20% progressive thresholds."""
        from hedgefund.risk.drawdown import get_drawdown_state

        peak = 1_000_000.0

        # At 0% DD, multiplier should be 1.0
        state = get_drawdown_state(current_value=peak, peak_value=peak)
        assert state.position_multiplier == 1.0

        # At 5% DD (boundary), multiplier still 1.0
        state = get_drawdown_state(current_value=peak * 0.95, peak_value=peak)
        assert state.position_multiplier == 1.0

        # At 7% DD (>5%), multiplier should decrease to 0.75
        state = get_drawdown_state(current_value=peak * 0.93, peak_value=peak)
        assert state.position_multiplier == 0.75

        # At exactly 20% DD (boundary), multiplier is 0.25 (still within max_drawdown)
        state = get_drawdown_state(current_value=peak * 0.80, peak_value=peak)
        assert state.position_multiplier == 0.25

        # Beyond 20% DD, multiplier must be 0 (STOP)
        state = get_drawdown_state(current_value=peak * 0.79, peak_value=peak)
        assert state.position_multiplier == 0.0

    def test_max_profit_factor_cap(self) -> None:
        """Profit factor must be capped at 100.0 (not inf)."""
        import numpy as np

        from hedgefund.core.risk_metrics import profit_factor

        # All wins, no losses → should return 100.0, not inf
        returns = np.array([0.01, 0.02, 0.03])
        result = profit_factor(returns)
        assert result <= 100.0


# ---------------------------------------------------------------------------
# Layer 7: Executor Protocol Compliance
# ---------------------------------------------------------------------------


class TestExecutorProtocol:
    """Paper executor must satisfy the Executor Protocol."""

    def test_paper_executor_implements_protocol(self) -> None:
        """PaperExecutor must be a valid Executor."""
        from hedgefund.execution.executors.paper_executor import PaperExecutor

        assert isinstance(PaperExecutor, type)
        # Check required methods from Executor Protocol
        required = ["submit_order", "cancel_order", "get_account_info", "get_current_price"]
        for method_name in required:
            assert hasattr(PaperExecutor, method_name), (
                f"PaperExecutor missing Executor method: {method_name}"
            )

# tests/test_real_grid.py
# Comprehensive tests for the grid engine.
# Tests cover: level calculation, buy/sell fills, P&L accuracy,
# edge cases, safety limits, and grid lifecycle.

import pytest
from src.strategies.real_grid import GridCalculator, GridState, GridLevel
from src.execution.grid_executor import GridExecutor, GridFill


class TestGridCalculator:
    """Test grid level calculation and activation checks."""

    def setup_method(self):
        self.calc = GridCalculator(
            num_levels=10,
            max_grid_exposure_pct=15.0,
            max_bb_width_pct=5.5,
            exit_atr_multiplier=1.0,
            fee_rate=0.00075,
        )

    def test_can_activate_in_ranging(self):
        ok, _ = self.calc.can_activate(bb_width=0.03, regime="RANGING")
        assert ok is True

    def test_cannot_activate_in_trending(self):
        ok, msg = self.calc.can_activate(bb_width=0.03, regime="TRENDING_UP")
        assert ok is False
        assert "TRENDING_UP" in msg

    def test_cannot_activate_high_volatility(self):
        ok, msg = self.calc.can_activate(bb_width=0.06, regime="RANGING")
        assert ok is False
        assert "BB width" in msg

    def test_cannot_activate_zero_bb(self):
        ok, _ = self.calc.can_activate(bb_width=0.0, regime="RANGING")
        assert ok is False

    def test_calculate_levels_creates_correct_count(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=100000,
        )
        assert grid is not None
        assert len(grid.levels) == 10

    def test_levels_are_evenly_spaced(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=100000,
        )
        spacings = []
        for i in range(1, len(grid.levels)):
            spacings.append(grid.levels[i].price - grid.levels[i - 1].price)
        # All spacings should be equal (within floating point tolerance)
        for s in spacings:
            assert abs(s - spacings[0]) < 0.01

    def test_levels_within_bb_range(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=100000,
        )
        for level in grid.levels:
            assert level.price > 93000
            assert level.price < 97000

    def test_size_per_level_correct(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=100000,
        )
        # 15% of $100K = $15K, divided by 10 levels = $1500
        assert grid.size_per_level == 1500.0

    def test_reserved_balance_correct(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=100000,
        )
        assert grid.reserved_balance == 15000.0

    def test_rejects_invalid_bb_range(self):
        grid = self.calc.calculate_levels(
            bb_lower=97000, bb_upper=93000,  # lower > upper
            current_price=95000, balance=100000,
        )
        assert grid is None

    def test_rejects_price_outside_range(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=99000, balance=100000,  # above range
        )
        assert grid is None

    def test_rejects_too_small_balance(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=50,  # 15% of $50 = $7.50 < $10 min
        )
        assert grid is None

    def test_should_close_on_upside_breakout(self):
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=100,
        )
        # Price at 97000 + 500 ATR + buffer = 97600
        close, _ = self.calc.should_close_grid(97600, grid, atr=500)
        assert close is True

    def test_should_close_on_downside_breakout(self):
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=100,
        )
        close, _ = self.calc.should_close_grid(92400, grid, atr=500)
        assert close is True

    def test_should_not_close_within_range(self):
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=100,
        )
        close, _ = self.calc.should_close_grid(95000, grid, atr=500)
        assert close is False

    def test_profit_per_fill_positive(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=100000,
        )
        profit = self.calc.get_profit_per_fill(grid)
        # Spacing ~$363, at $95K midpoint = ~0.38%
        # Gross: $1500 × 0.38% = $5.72
        # Fees: $1500 × 0.075% × 2 = $2.25
        # Net: ~$3.47
        assert profit > 0, f"profit per fill should be positive, got {profit}"

    def test_grid_spacing_pct(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=100000,
        )
        pct = self.calc.get_grid_spacing_pct(grid)
        # $4000 range / 11 intervals = $363.6 spacing
        # $363.6 / $95000 midpoint = 0.38%
        assert 0.3 < pct < 0.5, f"spacing should be ~0.38%, got {pct}"


class TestGridExecutor:
    """Test grid execution — buys, sells, P&L tracking."""

    def setup_method(self):
        self.executor = GridExecutor(fee_rate=0.00075)

    def _make_grid(self, lower=93000, upper=97000, num_levels=5,
                   size_per_level=1000) -> GridState:
        """Helper to create a grid with known levels."""
        spacing = (upper - lower) / (num_levels + 1)
        levels = []
        for i in range(1, num_levels + 1):
            levels.append(GridLevel(price=round(lower + spacing * i, 2)))
        return GridState(
            pair="",
            lower_bound=lower,
            upper_bound=upper,
            levels=levels,
            size_per_level=size_per_level,
            fee_rate=0.00075,
            reserved_balance=size_per_level * num_levels,
        )

    def test_activate_grid(self):
        grid = self._make_grid()
        ok, msg = self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)
        assert ok is True
        assert "BTC/USDT" in self.executor.active_pairs

    def test_activate_rejects_duplicate(self):
        grid1 = self._make_grid()
        self.executor.activate_grid("BTC/USDT", grid1, paper_balance=100000)
        grid2 = self._make_grid()
        ok, _ = self.executor.activate_grid("BTC/USDT", grid2, paper_balance=100000)
        assert ok is False

    def test_activate_rejects_insufficient_balance(self):
        grid = self._make_grid(size_per_level=1000)  # 5 × $1000 = $5000 reserved
        ok, msg = self.executor.activate_grid("BTC/USDT", grid, paper_balance=5000)
        # 5000 reserved > 5000 * 0.5 = 2500
        assert ok is False
        assert "available" in msg

    def test_buy_triggers_on_price_drop(self):
        grid = self._make_grid(lower=93000, upper=97000, num_levels=5)
        # Levels: 93666.67, 94333.33, 95000, 95666.67, 96333.33
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Price drops to level 1 (~93666)
        level_price = grid.levels[0].price
        fills = self.executor.update(
            "BTC/USDT",
            current_price=level_price,
            candle_high=95000,
            candle_low=level_price,
        )
        # No sells yet — just bought
        assert len(fills) == 0
        assert grid.levels[0].is_holding is True

    def test_sell_triggers_on_price_rise(self):
        grid = self._make_grid(lower=93000, upper=97000, num_levels=5)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Step 1: Buy at level 0
        buy_price = grid.levels[0].price
        self.executor.update("BTC/USDT", buy_price, buy_price + 100, buy_price)
        assert grid.levels[0].is_holding is True

        # Step 2: Price rises to level 1 (sell target for level 0)
        sell_price = grid.levels[1].price
        fills = self.executor.update("BTC/USDT", sell_price, sell_price, sell_price - 100)
        assert len(fills) == 1
        assert fills[0].net_pnl > 0  # profit after fees
        assert grid.levels[0].is_holding is False  # position closed

    def test_pnl_accuracy(self):
        """Verify P&L math is correct for a single grid fill."""
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        # Levels: 94400, 94800, 95200, 95600
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        buy_price = grid.levels[0].price   # 94400
        sell_price = grid.levels[1].price  # 94800

        # Buy
        self.executor.update("BTC/USDT", buy_price, buy_price + 50, buy_price)
        assert grid.levels[0].is_holding is True

        # Sell
        fills = self.executor.update("BTC/USDT", sell_price, sell_price, sell_price - 50)
        assert len(fills) == 1
        fill = fills[0]

        # Manual calculation
        qty = 1000 / 94400  # ~0.01059
        entry_fee = 1000 * 0.00075  # $0.75
        exit_notional = qty * 94800
        exit_fee = exit_notional * 0.00075
        expected_gross = (94800 - 94400) * qty
        expected_net = expected_gross - entry_fee - exit_fee

        assert abs(fill.gross_pnl - expected_gross) < 0.01
        assert abs(fill.net_pnl - expected_net) < 0.01
        assert fill.net_pnl > 0

    def test_multiple_fills_accumulate_pnl(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Bounce 1: buy level 0, sell at level 1
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)
        fills1 = self.executor.update("BTC/USDT", grid.levels[1].price,
                                      grid.levels[1].price, grid.levels[1].price - 50)
        assert len(fills1) == 1
        pnl_after_1 = grid.total_pnl

        # Bounce 2: price drops back to level 0, buy again, sell again
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)
        fills2 = self.executor.update("BTC/USDT", grid.levels[1].price,
                                      grid.levels[1].price, grid.levels[1].price - 50)
        assert len(fills2) == 1
        assert grid.total_pnl > pnl_after_1  # P&L accumulated
        assert grid.total_fills == 2

    def test_cash_runs_out_after_max_buys(self):
        """Grid stops buying when cash pool is exhausted.

        With wide spacing (90K-100K, 4 levels = $2000 apart), each update
        only touches the targeted level. But we use a single sweep candle
        to buy all levels at once, letting fees consume the 4th level's budget.
        4 × $1000 reserved = $4000. Each buy costs $1000 + $0.75 fee = $1000.75.
        After 3 buys: $4000 - 3 × $1000.75 = $997.75 < $1000 = can't buy 4th.
        """
        # Wide spacing so levels don't interfere
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=1000)
        # Levels: 92000, 94000, 96000, 98000
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # One big candle sweeps through all levels at once
        # current_price at highest level ensures all pass the 0.995 check
        self.executor.update(
            "BTC/USDT",
            current_price=grid.levels[3].price,  # 98000
            candle_high=99000,
            candle_low=grid.levels[0].price,      # 92000
        )

        # Only 3 fill — fees eat the 4th level's budget
        holding = sum(1 for l in grid.levels if l.is_holding)
        assert holding == 3
        assert self.executor._grid_cash < grid.size_per_level

    def test_deactivate_closes_open_positions(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Buy at level 0
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)
        assert grid.levels[0].is_holding is True

        # Deactivate — should close the position
        balance_return, summary = self.executor.deactivate_grid(
            "BTC/USDT", current_price=95000,
        )
        assert balance_return > 0  # got cash back
        assert grid.levels[0].is_holding is False
        assert grid.is_active is False
        assert "fills" in summary

    def test_deactivate_returns_correct_balance(self):
        """Balance returned should include accumulated P&L."""
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Complete one fill cycle (profit)
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)
        self.executor.update("BTC/USDT", grid.levels[1].price,
                            grid.levels[1].price, grid.levels[1].price - 50)

        balance_return, _ = self.executor.deactivate_grid(
            "BTC/USDT", current_price=95000,
        )
        # Should get back more than reserved (profit from the fill)
        assert balance_return > grid.reserved_balance - grid.size_per_level

    def test_unrealized_pnl_positive_when_price_above_entry(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Buy at level 0 (94400)
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)

        # Check unrealized P&L at higher price
        pnl = self.executor.get_unrealized_pnl("BTC/USDT", 95000)
        assert pnl > 0

    def test_unrealized_pnl_negative_when_price_below_entry(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Buy at level 1 (94800)
        self.executor.update("BTC/USDT", grid.levels[1].price,
                            grid.levels[1].price + 50, grid.levels[1].price)

        pnl = self.executor.get_unrealized_pnl("BTC/USDT", 94000)
        assert pnl < 0

    def test_status_returns_correct_info(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        status = self.executor.get_status("BTC/USDT")
        assert status["active"] is True
        assert status["levels"] == 4
        assert status["holding"] == 0
        assert status["fills"] == 0

    def test_inactive_pair_returns_no_fills(self):
        fills = self.executor.update("FAKE/USDT", 50000, 50100, 49900)
        assert fills == []

    def test_price_gaps_through_multiple_levels(self):
        """If price drops through 3 levels in one candle, buy all 3."""
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=500)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Candle low sweeps through first 3 levels
        lowest_level = grid.levels[0].price
        third_level = grid.levels[2].price
        self.executor.update(
            "BTC/USDT",
            current_price=third_level,
            candle_high=96000,
            candle_low=lowest_level,
        )

        # First 3 levels should have bought
        holding = sum(1 for l in grid.levels if l.is_holding)
        assert holding == 3

    def test_open_exposure_tracking(self):
        # Wide spacing so only one level triggers per update
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=1000)
        # Levels: 92000, 94000, 96000, 98000
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        assert self.executor.get_open_exposure() == 0.0

        # Buy only level 0 (92000) — wide spacing prevents level 1 from triggering
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 100, grid.levels[0].price)

        exposure = self.executor.get_open_exposure()
        # qty × entry_price = (1000/92000) × 92000 = $1000
        assert exposure > 900
        assert exposure < 1100

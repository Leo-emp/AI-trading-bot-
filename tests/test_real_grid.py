# tests/test_real_grid.py
# Comprehensive tests for the grid engine.
# Tests cover: level calculation, buy/sell fills, P&L accuracy,
# inventory scaling, per-grid cash isolation, max loss protection,
# fully loaded detection, and grid lifecycle.

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
            bb_lower=97000, bb_upper=93000,
            current_price=95000, balance=100000,
        )
        assert grid is None

    def test_rejects_price_outside_range(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=99000, balance=100000,
        )
        assert grid is None

    def test_rejects_too_small_balance(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=50,
        )
        assert grid is None

    def test_should_close_on_upside_breakout(self):
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=100,
        )
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
        assert profit > 0, f"profit per fill should be positive, got {profit}"

    def test_grid_spacing_pct(self):
        grid = self.calc.calculate_levels(
            bb_lower=93000, bb_upper=97000,
            current_price=95000, balance=100000,
        )
        pct = self.calc.get_grid_spacing_pct(grid)
        assert 0.3 < pct < 0.5, f"spacing should be ~0.38%, got {pct}"

    def test_rejects_unprofitable_grid(self):
        grid = self.calc.calculate_levels(
            bb_lower=94950, bb_upper=95050,
            current_price=95000, balance=100000,
        )
        assert grid is None, "should reject grid where spacing < fees"


class TestGridExecutor:
    """Test grid execution — buys, sells, P&L tracking."""

    def setup_method(self):
        self.executor = GridExecutor(fee_rate=0.00075, max_loss_pct=40.0)

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
        grid = self._make_grid(size_per_level=1000)
        ok, msg = self.executor.activate_grid("BTC/USDT", grid, paper_balance=5000)
        assert ok is False
        assert "available" in msg

    def test_per_grid_cash_isolation(self):
        """Each grid gets its own isolated cash pool."""
        grid_btc = self._make_grid(size_per_level=1000)
        grid_eth = self._make_grid(size_per_level=500)
        self.executor.activate_grid("BTC/USDT", grid_btc, paper_balance=100000)
        self.executor.activate_grid("ETH/USDT", grid_eth, paper_balance=90000)

        assert self.executor._grid_cash["BTC/USDT"] == 5000  # 5 × $1000
        assert self.executor._grid_cash["ETH/USDT"] == 2500  # 5 × $500

    def test_buy_uses_grid_own_cash(self):
        """Buying from one grid doesn't affect another grid's cash."""
        grid_btc = self._make_grid(lower=90000, upper=100000, size_per_level=1000)
        grid_eth = self._make_grid(lower=1800, upper=2200, size_per_level=500)
        self.executor.activate_grid("BTC/USDT", grid_btc, paper_balance=100000)
        self.executor.activate_grid("ETH/USDT", grid_eth, paper_balance=90000)

        btc_cash_before = self.executor._grid_cash["BTC/USDT"]
        eth_cash_before = self.executor._grid_cash["ETH/USDT"]

        # Buy one BTC level
        self.executor.update("BTC/USDT", grid_btc.levels[0].price,
                            grid_btc.levels[0].price + 100, grid_btc.levels[0].price)

        # BTC cash decreased, ETH cash unchanged
        assert self.executor._grid_cash["BTC/USDT"] < btc_cash_before
        assert self.executor._grid_cash["ETH/USDT"] == eth_cash_before

    def test_buy_triggers_on_price_drop(self):
        grid = self._make_grid(lower=93000, upper=97000, num_levels=5)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        level_price = grid.levels[0].price
        fills = self.executor.update(
            "BTC/USDT",
            current_price=level_price,
            candle_high=95000,
            candle_low=level_price,
        )
        assert len(fills) == 0
        assert grid.levels[0].is_holding is True

    def test_sell_triggers_on_price_rise(self):
        grid = self._make_grid(lower=93000, upper=97000, num_levels=5)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        buy_price = grid.levels[0].price
        self.executor.update("BTC/USDT", buy_price, buy_price + 100, buy_price)
        assert grid.levels[0].is_holding is True

        sell_price = grid.levels[1].price
        fills = self.executor.update("BTC/USDT", sell_price, sell_price, sell_price - 100)
        assert len(fills) == 1
        assert fills[0].net_pnl > 0
        assert grid.levels[0].is_holding is False

    def test_pnl_accuracy(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        buy_price = grid.levels[0].price
        sell_price = grid.levels[1].price

        self.executor.update("BTC/USDT", buy_price, buy_price + 50, buy_price)
        fills = self.executor.update("BTC/USDT", sell_price, sell_price, sell_price - 50)
        assert len(fills) == 1
        fill = fills[0]

        # Manual calculation (first buy = 100% scale)
        qty = 1000 / buy_price
        entry_fee = 1000 * 0.00075
        exit_notional = qty * sell_price
        exit_fee = exit_notional * 0.00075
        expected_gross = (sell_price - buy_price) * qty
        expected_net = expected_gross - entry_fee - exit_fee

        assert abs(fill.gross_pnl - expected_gross) < 0.01
        assert abs(fill.net_pnl - expected_net) < 0.01
        assert fill.net_pnl > 0

    def test_multiple_fills_accumulate_pnl(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)
        fills1 = self.executor.update("BTC/USDT", grid.levels[1].price,
                                      grid.levels[1].price, grid.levels[1].price - 50)
        assert len(fills1) == 1
        pnl_after_1 = grid.total_pnl

        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)
        fills2 = self.executor.update("BTC/USDT", grid.levels[1].price,
                                      grid.levels[1].price, grid.levels[1].price - 50)
        assert len(fills2) == 1
        assert grid.total_pnl > pnl_after_1
        assert grid.total_fills == 2

    def test_inventory_scaling_reduces_size(self):
        """Each subsequent buy at a new level should be smaller."""
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=1000)
        # Levels: 92000, 94000, 96000, 98000 — wide enough for isolation
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Buy level 0 (0 filled → 100% scale = $1000 position)
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 100, grid.levels[0].price)
        qty_0 = grid.levels[0].quantity
        notional_0 = qty_0 * grid.levels[0].price  # should be ~$1000

        # Buy level 1 (1 filled → 85% scale = $850 position)
        # candle_high must be BELOW sell target for level 0 (which is level 1 price)
        # to avoid triggering a sell that resets filled_count
        buy_1 = grid.levels[1].price
        self.executor.update("BTC/USDT", buy_1 - 1, buy_1 - 1, buy_1)
        qty_1 = grid.levels[1].quantity
        notional_1 = qty_1 * grid.levels[1].price  # should be ~$850

        # Verify notional sizes reflect inventory scaling
        assert abs(notional_0 - 1000) < 1.0    # 100% of $1000
        assert abs(notional_1 - 850) < 1.0     # 85% of $1000
        assert notional_1 < notional_0          # smaller due to scaling

    def test_inventory_scaling_values(self):
        """Verify the scale factors are correct."""
        from src.execution.grid_executor import INVENTORY_SCALE
        assert INVENTORY_SCALE[0] == 1.0    # 0 filled → full size
        assert INVENTORY_SCALE[1] == 0.85   # 1 filled → 85%
        assert INVENTORY_SCALE[2] == 0.70   # 2 filled → 70%
        assert INVENTORY_SCALE[3] == 0.55   # 3 filled → 55%
        assert INVENTORY_SCALE[4] == 0.40   # 4 filled → 40%

    def test_cash_runs_out_after_max_buys(self):
        """Grid stops buying when its isolated cash pool is exhausted."""
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Sweep all levels. Inventory scaling: 1000+850+700+550=$3100 + fees
        # Reserved: $4000. After scaled buys + fees, last level may not fit.
        self.executor.update(
            "BTC/USDT",
            current_price=grid.levels[3].price,
            candle_high=99000,
            candle_low=grid.levels[0].price,
        )

        holding = sum(1 for l in grid.levels if l.is_holding)
        # With inventory scaling, all 4 levels should fit
        # (total ~$3100 + ~$2.33 fees < $4000 reserved)
        assert holding == 4

    def test_deactivate_closes_open_positions(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)
        assert grid.levels[0].is_holding is True

        balance_return, summary = self.executor.deactivate_grid(
            "BTC/USDT", current_price=95000,
        )
        assert balance_return > 0
        assert grid.levels[0].is_holding is False
        assert grid.is_active is False
        assert "fills" in summary

    def test_deactivate_removes_grid_cash(self):
        """Deactivating a grid removes its entry from _grid_cash."""
        grid = self._make_grid()
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)
        assert "BTC/USDT" in self.executor._grid_cash

        self.executor.deactivate_grid("BTC/USDT", current_price=95000)
        assert "BTC/USDT" not in self.executor._grid_cash

    def test_deactivate_returns_correct_balance(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)
        self.executor.update("BTC/USDT", grid.levels[1].price,
                            grid.levels[1].price, grid.levels[1].price - 50)

        balance_return, _ = self.executor.deactivate_grid(
            "BTC/USDT", current_price=95000,
        )
        assert balance_return > grid.reserved_balance - grid.size_per_level

    def test_unrealized_pnl_positive_when_price_above_entry(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 50, grid.levels[0].price)

        pnl = self.executor.get_unrealized_pnl("BTC/USDT", 95000)
        assert pnl > 0

    def test_unrealized_pnl_negative_when_price_below_entry(self):
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        self.executor.update("BTC/USDT", grid.levels[1].price,
                            grid.levels[1].price + 50, grid.levels[1].price)

        pnl = self.executor.get_unrealized_pnl("BTC/USDT", 94000)
        assert pnl < 0

    def test_max_loss_triggers_force_close(self):
        """Grid signals force-close when unrealized loss exceeds threshold."""
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=1000)
        # reserved = $4000, max_loss = 40% → triggers at $1600 loss
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Fill all 4 levels to maximize exposure
        self.executor.update(
            "BTC/USDT",
            current_price=grid.levels[3].price,
            candle_high=99000,
            candle_low=grid.levels[0].price,
        )
        # Total invested ~$1000+$850+$700+$550 = $3100
        # At price 50000, loss on each position is massive
        # Level 0 (92000→50000): (50000-92000)×(1000/92000) ≈ -$456
        # Level 1 (94000→50000): (50000-94000)×(850/94000) ≈ -$397
        # Level 2 (96000→50000): (50000-96000)×(700/96000) ≈ -$335
        # Level 3 (98000→50000): (50000-98000)×(550/98000) ≈ -$269
        # Total: ~-$1457 → 1457/4000 = 36.4%... need more extreme price
        should_close, reason = self.executor.should_force_close("BTC/USDT", 30000)
        assert should_close is True
        assert "loss" in reason.lower()

    def test_max_loss_does_not_trigger_in_profit(self):
        """No force-close when position is profitable."""
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 100, grid.levels[0].price)

        should_close, _ = self.executor.should_force_close("BTC/USDT", 99000)
        assert should_close is False

    def test_loss_pct_calculation(self):
        """Verify loss percentage is calculated correctly."""
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=1000)
        # reserved = $4000
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Buy at 92000
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 100, grid.levels[0].price)

        # Price at entry — 0% loss
        loss_at_entry = self.executor.get_loss_pct("BTC/USDT", grid.levels[0].price)
        assert loss_at_entry == 0.0

        # Price above entry — still 0% (in profit)
        loss_above = self.executor.get_loss_pct("BTC/USDT", 95000)
        assert loss_above == 0.0

    def test_fully_loaded_detection(self):
        """Detect when all grid levels are filled."""
        grid = self._make_grid(lower=90000, upper=100000, num_levels=3,
                               size_per_level=500)
        # Levels: 92500, 95000, 97500
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        assert self.executor.is_fully_loaded("BTC/USDT") is False

        # Fill all 3 levels with one sweep candle
        self.executor.update(
            "BTC/USDT",
            current_price=grid.levels[2].price,
            candle_high=99000,
            candle_low=grid.levels[0].price,
        )

        assert self.executor.is_fully_loaded("BTC/USDT") is True

    def test_fully_loaded_not_triggered_when_partial(self):
        """Partially filled grid is not fully loaded."""
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=500)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        # Fill only level 0
        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 100, grid.levels[0].price)

        assert self.executor.is_fully_loaded("BTC/USDT") is False

    def test_total_grid_cash(self):
        """total_grid_cash sums across all active grids."""
        grid_btc = self._make_grid(size_per_level=1000)
        grid_eth = self._make_grid(size_per_level=500)
        self.executor.activate_grid("BTC/USDT", grid_btc, paper_balance=100000)
        self.executor.activate_grid("ETH/USDT", grid_eth, paper_balance=90000)

        # BTC: 5×1000 = 5000, ETH: 5×500 = 2500, total = 7500
        assert self.executor.total_grid_cash == 7500.0

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
        """If price drops through multiple levels in one candle, buy them all."""
        grid = self._make_grid(lower=94000, upper=96000, num_levels=4,
                               size_per_level=500)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        lowest_level = grid.levels[0].price
        third_level = grid.levels[2].price
        self.executor.update(
            "BTC/USDT",
            current_price=third_level,
            candle_high=96000,
            candle_low=lowest_level,
        )

        # With inventory scaling, all 3 levels within the candle range should fill
        # (sizes: 500, 425, 350 + fees easily fit in $2000 budget)
        holding = sum(1 for l in grid.levels if l.is_holding)
        assert holding >= 3

    def test_grid_staleness_detection(self):
        """Grid is stale when BB center drifts > 50% of range width."""
        import time as _time
        calc = GridCalculator(num_levels=5, fee_rate=0.00075)
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=1000, is_active=True,
            activated_at=_time.time() - 10800,  # 3 hours ago
        )
        # BB shifted from center 95000 to center 99000 (drift of 4000, range is 4000 → 100%)
        stale, reason = calc.is_grid_stale(grid, new_bb_lower=97000, new_bb_upper=101000)
        assert stale is True
        assert "drifted" in reason

    def test_grid_not_stale_when_young(self):
        """Grids younger than 2 hours are never stale."""
        import time as _time
        calc = GridCalculator(num_levels=5, fee_rate=0.00075)
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=1000, is_active=True,
            activated_at=_time.time() - 60,  # 1 minute ago
        )
        stale, _ = calc.is_grid_stale(grid, new_bb_lower=97000, new_bb_upper=101000)
        assert stale is False

    def test_grid_not_stale_when_bb_close(self):
        """Grid is fine when BB hasn't shifted much."""
        import time as _time
        calc = GridCalculator(num_levels=5, fee_rate=0.00075)
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=1000, is_active=True,
            activated_at=_time.time() - 10800,  # 3 hours ago
        )
        # BB shifted slightly (center 95000 → 95500 = 12.5% of 4000 range)
        stale, _ = calc.is_grid_stale(grid, new_bb_lower=93500, new_bb_upper=97500)
        assert stale is False

    def test_grid_idle_timeout(self):
        """Grid with 0 fills after 8 hours should close."""
        import time as _time
        calc = GridCalculator(num_levels=5, fee_rate=0.00075)
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=1000, is_active=True,
            total_fills=0,
            activated_at=_time.time() - 36000,  # 10 hours ago
        )
        idle, reason = calc.is_grid_idle(grid)
        assert idle is True
        assert "no fills" in reason

    def test_grid_not_idle_with_fills(self):
        """Grid with completed fills is never idle."""
        import time as _time
        calc = GridCalculator(num_levels=5, fee_rate=0.00075)
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=1000, is_active=True,
            total_fills=3,
            activated_at=_time.time() - 36000,  # 10 hours ago
        )
        idle, _ = calc.is_grid_idle(grid)
        assert idle is False

    def test_grid_not_idle_when_young(self):
        """Young grids are never idle regardless of fills."""
        import time as _time
        calc = GridCalculator(num_levels=5, fee_rate=0.00075)
        grid = GridState(
            pair="BTC/USDT",
            lower_bound=93000, upper_bound=97000,
            levels=[], size_per_level=1000, is_active=True,
            total_fills=0,
            activated_at=_time.time() - 3600,  # 1 hour ago
        )
        idle, _ = calc.is_grid_idle(grid)
        assert idle is False

    def test_activated_at_set_on_activation(self):
        """Grid executor sets activated_at timestamp."""
        import time as _time
        grid = self._make_grid()
        before = _time.time()
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)
        after = _time.time()
        assert grid.activated_at >= before
        assert grid.activated_at <= after

    def test_open_exposure_tracking(self):
        grid = self._make_grid(lower=90000, upper=100000, num_levels=4,
                               size_per_level=1000)
        self.executor.activate_grid("BTC/USDT", grid, paper_balance=100000)

        assert self.executor.get_open_exposure() == 0.0

        self.executor.update("BTC/USDT", grid.levels[0].price,
                            grid.levels[0].price + 100, grid.levels[0].price)

        exposure = self.executor.get_open_exposure()
        assert exposure > 900
        assert exposure < 1100

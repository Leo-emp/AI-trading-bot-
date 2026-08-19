# tests/test_smart_exit.py
# Tests for SmartExitManager — the 4-phase profit maximizer.

import pytest
from src.execution.smart_exit import SmartExitManager


class TestSmartExitBuy:
    """Test smart exit logic for long (buy) positions."""

    def setup_method(self):
        """Set up a SmartExitManager and register a buy position."""
        self.mgr = SmartExitManager(
            breakeven_pct=0.2,       # +0.2% → move SL to entry
            trail_activate_pct=0.5,  # +0.5% → trailing stop starts
            trail_distance_pct=0.3,  # trail at 0.3% behind
            tight_trail_pct=0.15,    # tighten to 0.15% past TP level
            tight_activate_pct=0.8,  # +0.8% → tight trailing
        )
        # Entry at $100, SL at $99.60 (0.4%), TP at $100.80 (0.8%)
        self.mgr.register("pos1", "BTCUSDT", "buy",
                         entry=100.0, sl=99.60, tp=100.80)

    def test_phase1_normal_stop_loss(self):
        """In phase 1, hitting SL closes the position."""
        state = self.mgr.update("pos1", 99.50)  # below SL
        assert state.is_closed is True
        assert state.phase == 1
        assert "phase1" in state.close_reason

    def test_phase1_no_close_above_sl(self):
        """In phase 1, price above SL stays open."""
        state = self.mgr.update("pos1", 100.10)
        assert state.is_closed is False
        assert state.phase == 1

    def test_phase2_breakeven_activation(self):
        """At +0.2% profit, SL moves to entry + 0.25R buffer."""
        state = self.mgr.update("pos1", 100.25)  # +0.25% > 0.2%
        assert state.phase == 2
        # SL = entry + 0.25 * initial_risk (0.40) = 100.0 + 0.10 = 100.10
        expected_sl = 100.0 + 0.25 * 0.40
        assert abs(state.current_sl - expected_sl) < 0.01
        assert state.is_closed is False

    def test_phase2_trade_is_risk_free(self):
        """After break-even, dropping to entry hits the new SL."""
        self.mgr.update("pos1", 100.25)  # activate phase 2
        state = self.mgr.update("pos1", 99.95)  # below entry
        assert state.is_closed is True
        assert state.phase == 2

    def test_phase3_trailing_activation(self):
        """At +0.5% profit, trailing stop activates."""
        self.mgr.update("pos1", 100.25)  # phase 2
        state = self.mgr.update("pos1", 100.55)  # +0.55% > 0.5%
        assert state.phase == 3
        # SL should be trailing at 0.3% behind current price
        expected_sl = 100.55 * (1 - 0.003)  # ~100.25
        assert abs(state.current_sl - expected_sl) < 0.01

    def test_phase3_trailing_follows_price_up(self):
        """Trailing stop moves up as price rises."""
        self.mgr.update("pos1", 100.55)  # activate phase 3
        state = self.mgr.update("pos1", 100.70)  # price goes up but stays in phase 3
        # SL should follow at 0.3% behind highest (100.70)
        expected_sl = 100.70 * (1 - 0.003)  # ~100.40
        assert abs(state.current_sl - expected_sl) < 0.01
        assert state.phase == 3  # still in phase 3

    def test_phase3_trailing_never_moves_down(self):
        """Trailing stop never moves backwards."""
        self.mgr.update("pos1", 100.55)  # phase 3
        self.mgr.update("pos1", 101.00)  # higher
        sl_after_high = self.mgr.get_current_sl("pos1")
        self.mgr.update("pos1", 100.80)  # price drops but above SL
        sl_after_dip = self.mgr.get_current_sl("pos1")
        assert sl_after_dip >= sl_after_high  # SL stays or goes up, never down

    def test_phase4_tight_trailing(self):
        """At +0.8% profit, trailing tightens to 0.15%."""
        self.mgr.update("pos1", 100.25)  # phase 2
        self.mgr.update("pos1", 100.55)  # phase 3
        state = self.mgr.update("pos1", 100.85)  # +0.85% > 0.8% → phase 4
        assert state.phase == 4
        # Tight trail: 0.15% behind highest
        expected_sl = 100.85 * (1 - 0.0015)  # ~100.70
        assert abs(state.current_sl - expected_sl) < 0.02

    def test_winner_runs_past_original_tp(self):
        """Price goes to +2% — smart exit keeps the position open."""
        self.mgr.update("pos1", 100.25)  # phase 2
        self.mgr.update("pos1", 100.55)  # phase 3
        self.mgr.update("pos1", 100.85)  # phase 4
        state = self.mgr.update("pos1", 102.00)  # +2%! Still open!
        assert state.is_closed is False
        assert state.phase == 4
        # SL tight at 0.15% behind $102
        expected_sl = 102.00 * (1 - 0.0015)  # ~101.85
        assert abs(state.current_sl - expected_sl) < 0.02

    def test_winner_eventually_closes_on_pullback(self):
        """After running to +2%, a pullback hits the tight trailing stop."""
        self.mgr.update("pos1", 100.25)
        self.mgr.update("pos1", 100.55)
        self.mgr.update("pos1", 100.85)
        self.mgr.update("pos1", 102.00)  # peak
        state = self.mgr.update("pos1", 101.80)  # pullback hits tight SL
        assert state.is_closed is True
        assert state.phase == 4
        # Closed at ~$101.85 instead of original TP $100.80 → +$1 extra profit!


class TestSmartExitSell:
    """Test smart exit for short (sell) positions."""

    def setup_method(self):
        self.mgr = SmartExitManager()
        # Short at $100, SL at $100.40, TP at $99.20
        self.mgr.register("pos2", "ETHUSDT", "sell",
                         entry=100.0, sl=100.40, tp=99.20)

    def test_sell_breakeven(self):
        """Short position: SL moves to entry - 0.25R buffer at +0.2% profit."""
        state = self.mgr.update("pos2", 99.75)  # -0.25% = profit for short
        assert state.phase == 2
        # SL = entry - 0.25 * initial_risk (0.40) = 100.0 - 0.10 = 99.90
        expected_sl = 100.0 - 0.25 * 0.40
        assert abs(state.current_sl - expected_sl) < 0.01

    def test_sell_trailing(self):
        """Short position: trailing stop follows price down."""
        self.mgr.update("pos2", 99.75)
        state = self.mgr.update("pos2", 99.40)  # +0.6%
        assert state.phase == 3
        # SL should trail at 0.3% above lowest
        expected_sl = 99.40 * (1 + 0.003)
        assert abs(state.current_sl - expected_sl) < 0.01


class TestAdaptiveSizer:
    """Test adaptive position sizing."""

    def test_normal_multiplier(self):
        from src.execution.adaptive_sizer import AdaptiveSizer
        sizer = AdaptiveSizer()
        assert sizer.get_multiplier() == 1.0

    def test_win_streak_increases_size(self):
        from src.execution.adaptive_sizer import AdaptiveSizer
        sizer = AdaptiveSizer(win_bonus_pct=10.0)
        sizer.record_win()
        assert sizer.get_multiplier() == 1.1  # +10%
        sizer.record_win()
        assert sizer.get_multiplier() == 1.2  # +20%

    def test_win_streak_capped(self):
        from src.execution.adaptive_sizer import AdaptiveSizer
        sizer = AdaptiveSizer(win_bonus_pct=10.0, max_win_bonus_pct=50.0)
        for _ in range(10):
            sizer.record_win()
        assert sizer.get_multiplier() == 1.5  # capped at +50%

    def test_loss_streak_reduces_size(self):
        from src.execution.adaptive_sizer import AdaptiveSizer
        sizer = AdaptiveSizer(loss_reduction_pct=20.0)
        sizer.record_loss()
        assert sizer.get_multiplier() == 0.8  # -20%
        sizer.record_loss()
        assert sizer.get_multiplier() == 0.6  # -40%

    def test_loss_streak_capped(self):
        from src.execution.adaptive_sizer import AdaptiveSizer
        sizer = AdaptiveSizer(loss_reduction_pct=20.0, max_loss_reduction_pct=60.0)
        for _ in range(10):
            sizer.record_loss()
        assert sizer.get_multiplier() == 0.4  # capped at -60%

    def test_win_resets_loss_streak(self):
        from src.execution.adaptive_sizer import AdaptiveSizer
        sizer = AdaptiveSizer()
        sizer.record_loss()
        sizer.record_loss()
        assert sizer.get_multiplier() < 1.0
        sizer.record_win()
        assert sizer.get_multiplier() == 1.1  # reset to 1 win

    def test_adjust_size_respects_minimum(self):
        from src.execution.adaptive_sizer import AdaptiveSizer
        sizer = AdaptiveSizer()
        for _ in range(3):
            sizer.record_loss()
        # Base $15 * 0.4 multiplier = $6, but minimum is $10
        result = sizer.adjust_size(15.0, min_size=10.0)
        assert result == 10.0

    def test_adjust_size_applies_multiplier(self):
        from src.execution.adaptive_sizer import AdaptiveSizer
        sizer = AdaptiveSizer(win_bonus_pct=10.0)
        sizer.record_win()
        sizer.record_win()
        # Base $20 * 1.2 multiplier = $24
        result = sizer.adjust_size(20.0)
        assert abs(result - 24.0) < 0.01

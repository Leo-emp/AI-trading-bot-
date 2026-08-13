# backtest/report.py
# Prints and saves backtest performance reports.

from backtest.engine import BacktestResult


class BacktestReport:
    """Formats and displays backtest results."""

    @staticmethod
    def print_summary(result: BacktestResult, strategy_name: str = ""):
        """Print a human-readable performance summary."""
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS: {strategy_name}")
        print(f"{'='*60}")
        print(f"  Total trades:     {result.total_trades}")
        print(f"  Win rate:         {result.win_rate*100:.1f}%")
        print(f"  Net P&L:          ${result.net_pnl:.2f}")
        print(f"  Total fees:       ${result.total_fees:.4f}")
        print(f"  Sharpe ratio:     {result.sharpe_ratio:.3f}")
        print(f"  Max drawdown:     {result.max_drawdown_pct:.1f}%")
        print(f"  Final balance:    ${result.final_balance:.2f}")
        print(f"{'='*60}")

        # Pass/fail criteria
        passed = (
            result.win_rate >= 0.55
            and result.net_pnl > 0
            and result.sharpe_ratio > 1.0
            and result.max_drawdown_pct < 15
        )
        status = "PASSED" if passed else "FAILED"
        print(f"  Go-live gate:     {status}")
        if not passed:
            if result.win_rate < 0.55:
                print(f"    - Win rate {result.win_rate*100:.1f}% < 55% required")
            if result.net_pnl <= 0:
                print(f"    - Net P&L negative (${result.net_pnl:.2f})")
            if result.sharpe_ratio <= 1.0:
                print(f"    - Sharpe {result.sharpe_ratio:.3f} <= 1.0 required")
            if result.max_drawdown_pct >= 15:
                print(f"    - Max drawdown {result.max_drawdown_pct:.1f}% >= 15% limit")
        print()

    @staticmethod
    def save_report(result: BacktestResult, path: str):
        """Save results to a text file."""
        with open(path, "w") as f:
            f.write(f"Total trades: {result.total_trades}\n")
            f.write(f"Win rate: {result.win_rate*100:.1f}%\n")
            f.write(f"Net P&L: ${result.net_pnl:.2f}\n")
            f.write(f"Total fees: ${result.total_fees:.4f}\n")
            f.write(f"Sharpe ratio: {result.sharpe_ratio:.3f}\n")
            f.write(f"Max drawdown: {result.max_drawdown_pct:.1f}%\n")
            f.write(f"Final balance: ${result.final_balance:.2f}\n")

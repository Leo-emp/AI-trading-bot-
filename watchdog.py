# watchdog.py
# Process monitor that auto-restarts the trading bot if it crashes.
# Uses exponential backoff to prevent rapid restart loops.
#
# Usage:
#   python watchdog.py              → Monitor paper trading
#   python watchdog.py --live       → Monitor live trading (be careful!)
#
# The watchdog:
# - Starts main.py as a subprocess
# - If it exits with non-zero code, restarts after a delay
# - Exponential backoff: 10s → 20s → 40s → 80s → ... (caps at 300s)
# - Resets backoff after 5 minutes of stable running
# - Gives up after 20 consecutive crashes
#
# This is Layer 5 of the protection system (Infrastructure).

import subprocess
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Configuration ---
MAX_CONSECUTIVE_CRASHES = 20     # give up after this many
INITIAL_BACKOFF_SECONDS = 10     # first restart delay
MAX_BACKOFF_SECONDS = 300        # cap at 5 minutes
STABLE_THRESHOLD_SECONDS = 300   # reset backoff after 5 min stable


def run_watchdog():
    """Monitor and auto-restart the trading bot."""
    # Build the command to run
    args = sys.argv[1:]  # pass through CLI args (--live, --backtest, etc.)
    cmd = [sys.executable, "main.py"] + args

    consecutive_crashes = 0
    backoff = INITIAL_BACKOFF_SECONDS

    logging.info("Watchdog started. Monitoring: %s", " ".join(cmd))
    logging.info("Max consecutive crashes before giving up: %d", MAX_CONSECUTIVE_CRASHES)

    while consecutive_crashes < MAX_CONSECUTIVE_CRASHES:
        start_time = time.time()
        logging.info("Starting trading bot (attempt %d)...", consecutive_crashes + 1)

        try:
            # Run the bot as a subprocess
            result = subprocess.run(cmd, check=False)
            exit_code = result.returncode
        except KeyboardInterrupt:
            logging.info("Watchdog interrupted by user. Shutting down.")
            break
        except Exception as e:
            logging.error("Failed to start subprocess: %s", e)
            exit_code = 1

        runtime = time.time() - start_time

        # Clean exit (exit code 0) — user stopped it intentionally
        if exit_code == 0:
            logging.info("Bot exited cleanly (code 0). Watchdog stopping.")
            break

        # Crash detected
        consecutive_crashes += 1
        logging.warning(
            "Bot crashed! Exit code: %d, Runtime: %.0fs, Crash #%d/%d",
            exit_code, runtime, consecutive_crashes, MAX_CONSECUTIVE_CRASHES,
        )

        # Reset backoff if the bot ran for a while (it was stable)
        if runtime > STABLE_THRESHOLD_SECONDS:
            logging.info("Bot was stable for %.0fs — resetting backoff", runtime)
            backoff = INITIAL_BACKOFF_SECONDS
            consecutive_crashes = 1  # reset crash counter too

        logging.info("Restarting in %d seconds...", backoff)
        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            logging.info("Watchdog interrupted during backoff. Shutting down.")
            break

        # Exponential backoff (double each time, cap at max)
        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

    if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
        logging.critical(
            "GIVING UP: %d consecutive crashes. Bot needs manual intervention.",
            MAX_CONSECUTIVE_CRASHES,
        )
        sys.exit(1)


if __name__ == "__main__":
    run_watchdog()

#!/usr/bin/env python3
"""Force-close all open positions at current market prices.

Run on the server: python3 scripts/force_close.py
Closes old positions so the bot can reopen with correct leverage/fees.
"""
import json
import time
import os
import sys
import urllib.request

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "paper_state.json")
FEE_RATE = 0.0002  # Binance futures maker rate

def main():
    # Get current prices from Binance
    url = "https://api.binance.com/api/v3/ticker/price"
    data = json.loads(urllib.request.urlopen(url).read())
    prices = {d["symbol"]: float(d["price"]) for d in data}

    with open(STATE_FILE) as f:
        state = json.load(f)

    if not state["positions"]:
        print("No open positions to close.")
        return

    print("=== FORCE CLOSING %d POSITIONS ===" % len(state["positions"]))
    total_pnl = 0

    for pos in state["positions"]:
        pair = pos["pair"]
        symbol = pair.replace("/", "")
        current_price = prices.get(symbol, pos["entry_price"])
        entry = pos["entry_price"]
        qty = pos["quantity"]

        if pos["side"] == "buy":
            gross_pnl = (current_price - entry) * qty
        else:
            gross_pnl = (entry - current_price) * qty

        exit_fee = current_price * qty * FEE_RATE
        net_pnl = gross_pnl - pos.get("entry_fee", 0) - exit_fee

        margin = pos.get("margin_locked", entry * qty)
        balance_return = margin + gross_pnl - exit_fee
        if balance_return < 0:
            balance_return = 0

        state["balance"] += balance_return
        state["total_trades"] += 1
        if net_pnl > 0:
            state["total_wins"] += 1

        total_pnl += net_pnl
        print("  %s: entry=$%.4f now=$%.4f P&L=$%.2f" % (
            pair, entry, current_price, net_pnl))

    state["positions"] = []
    state["saved_at"] = time.time()

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print()
    print("Total close P&L: $%.2f" % total_pnl)
    print("New balance: $%.2f" % state["balance"])
    print("Ready for fresh trades with correct leverage + fees.")

if __name__ == "__main__":
    main()

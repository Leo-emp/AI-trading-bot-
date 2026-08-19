#!/usr/bin/env python3
# dashboard.py — Real-time trading dashboard for the AI Trading Bot.
#
# Serves a professional dark-themed web dashboard with:
# - Live market prices from Binance WebSocket (browser-direct)
# - Open positions, trade history, equity curve from the bot
# - Real-time log streaming
#
# Run:   python dashboard.py
# Access: http://<server-ip>:8080
#
# Architecture designed for easy Vercel migration:
# - dashboard.html is self-contained (inline CSS/JS)
# - API endpoints are clean REST + WebSocket
# - Just deploy the HTML to Vercel and point API calls to Oracle

import asyncio
import json
import os
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

import aiosqlite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logger = logging.getLogger("dashboard")

# --- Configuration ---
# All paths relative to this file (project root)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "paper_state.json"
DB_PATH = BASE_DIR / "trading_bot.db"
LOG_FILE = BASE_DIR / "trading.log"
HTML_FILE = BASE_DIR / "dashboard.html"
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))

# --- FastAPI App ---
app = FastAPI(title="AI Trading Bot Dashboard")

# CORS — allows a Vercel frontend to call this API later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_paper_state() -> dict:
    """Read the bot's current state from paper_state.json.

    This file is written by the bot every time a trade opens/closes.
    Contains: balance, positions, trade counts.
    """
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "balance": 0,
            "initial_balance": 100000,
            "total_trades": 0,
            "total_wins": 0,
            "positions": [],
        }


async def get_recent_trades(limit: int = 50) -> list[dict]:
    """Query recent closed trades from the SQLite database.

    Returns most recent trades first, with full P&L breakdown.
    """
    try:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error("DB query failed: %s", e)
        return []


async def get_portfolio_snapshots(hours: int = 72) -> list[dict]:
    """Get portfolio snapshots for the equity curve chart.

    Returns hourly snapshots from the last 72 hours.
    """
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM portfolio_snapshots "
                "WHERE timestamp > ? ORDER BY timestamp ASC",
                (cutoff,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error("Snapshot query failed: %s", e)
        return []


async def get_bot_data() -> dict:
    """Assemble all bot data into a single payload for the dashboard.

    Combines paper_state.json (live state) with database queries
    (trade history, equity curve). This is what the WebSocket pushes.
    """
    state = read_paper_state()
    trades = await get_recent_trades()
    snapshots = await get_portfolio_snapshots()

    # Calculate derived metrics
    initial = state.get("initial_balance", 100000)
    balance = state.get("balance", 0)
    total_trades = state.get("total_trades", 0)
    total_wins = state.get("total_wins", 0)
    positions = state.get("positions", [])
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    return {
        "balance": round(balance, 2),
        "initial_balance": round(initial, 2),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": round(win_rate, 1),
        "positions": positions,
        "recent_trades": trades,
        "snapshots": snapshots,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --- HTTP Routes ---

@app.get("/")
async def index():
    """Serve the dashboard HTML page."""
    try:
        html = HTML_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        html = "<h1>dashboard.html not found — place it next to dashboard.py</h1>"
    return HTMLResponse(html)


@app.get("/api/data")
async def api_data():
    """REST endpoint for initial data load (also useful for Vercel migration)."""
    return await get_bot_data()


# --- WebSocket Endpoints ---

@app.websocket("/ws/bot")
async def bot_websocket(websocket: WebSocket):
    """Stream bot data updates every 5 seconds.

    The frontend uses this for: positions, equity, trades, snapshots.
    Market prices come separately from Binance (browser-direct).
    """
    await websocket.accept()
    try:
        while True:
            data = await get_bot_data()
            await websocket.send_json(data)
            # 5-second refresh — fast enough for a dashboard,
            # slow enough to not hammer the DB
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Bot WebSocket error: %s", e)


@app.websocket("/ws/logs")
async def logs_websocket(websocket: WebSocket):
    """Stream trading.log lines in real time.

    Sends the last 200 lines on connect, then tails new lines.
    Color coding is done client-side based on log content.
    """
    await websocket.accept()
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                # Send last 200 lines on initial connect
                lines = f.readlines()
                initial = lines[-200:] if len(lines) > 200 else lines
                for line in initial:
                    await websocket.send_text(line.rstrip())

                # Tail new lines as they appear
                while True:
                    line = f.readline()
                    if line:
                        await websocket.send_text(line.rstrip())
                    else:
                        await asyncio.sleep(1)
        else:
            await websocket.send_text("[Dashboard] Log file not found")
            while True:
                await asyncio.sleep(10)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Log WebSocket error: %s", e)


# --- Entry Point ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    logger.info("Starting dashboard on http://0.0.0.0:%d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

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
import secrets
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

import aiosqlite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
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

# --- Auth ---
# Set DASHBOARD_TOKEN env var to require ?token=xxx on all requests.
# Without it, the dashboard is open (protected only by Oracle security list).
DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")
# Allowed origins for WebSocket and CORS (server's own origin always allowed)
ALLOWED_ORIGINS = {
    f"http://localhost:{PORT}",
    f"http://127.0.0.1:{PORT}",
    f"http://161.118.199.141:{PORT}",
}
# Add Vercel frontend origin if configured
_vercel_origin = os.environ.get("DASHBOARD_CORS_ORIGIN", "")
if _vercel_origin:
    ALLOWED_ORIGINS.add(_vercel_origin)

# --- FastAPI App ---
app = FastAPI(title="AI Trading Bot Dashboard")

# CORS — restricted to known origins, no credentials needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def verify_token(token: str) -> bool:
    """Check if the provided token matches DASHBOARD_TOKEN.
    If no token is configured, all requests pass (open mode).
    """
    if not DASHBOARD_TOKEN:
        return True
    return secrets.compare_digest(token, DASHBOARD_TOKEN)


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

    # Grid reserved cash — deducted from balance for grid levels but
    # not visible as positions. Must be added back to equity.
    grid_reserved = state.get("grid_reserved", 0)

    return {
        "balance": round(balance, 2),
        "initial_balance": round(initial, 2),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": round(win_rate, 1),
        "positions": positions,
        "grid_reserved": round(grid_reserved, 2),
        "recent_trades": trades,
        "snapshots": snapshots,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --- HTTP Routes ---

@app.get("/")
async def index(token: str = Query("")):
    """Serve the dashboard HTML page (token-gated if DASHBOARD_TOKEN is set)."""
    if not verify_token(token):
        return HTMLResponse("<h1>Unauthorized — add ?token=xxx</h1>", status_code=401)
    try:
        html = HTML_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        html = "<h1>dashboard.html not found — place it next to dashboard.py</h1>"
    # No-cache so browser always loads latest dashboard code
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/api/data")
async def api_data(token: str = Query("")):
    """REST endpoint for initial data load (token-gated)."""
    if not verify_token(token):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await get_bot_data()


# --- WebSocket Endpoints ---

def _check_ws_origin(websocket: WebSocket) -> bool:
    """Validate WebSocket origin header against allowlist."""
    origin = websocket.headers.get("origin", "")
    if not origin:
        return True
    return origin in ALLOWED_ORIGINS


@app.websocket("/ws/bot")
async def bot_websocket(websocket: WebSocket, token: str = Query("")):
    """Stream bot data updates every 5 seconds.

    The frontend uses this for: positions, equity, trades, snapshots.
    Market prices come separately from Binance (browser-direct).
    """
    if not verify_token(token) or not _check_ws_origin(websocket):
        await websocket.close(code=1008)
        return
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
async def logs_websocket(websocket: WebSocket, token: str = Query("")):
    """Stream trading.log lines in real time.

    Sends the last 200 lines on connect, then tails new lines.
    Color coding is done client-side based on log content.
    """
    if not verify_token(token) or not _check_ws_origin(websocket):
        await websocket.close(code=1008)
        return
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

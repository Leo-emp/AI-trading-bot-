#!/bin/bash
# setup_server.sh
# One-command server setup for the AI Trading Bot.
# Run this on your Oracle Cloud VM after cloning the repo.
#
# Usage:
#   chmod +x deploy/setup_server.sh
#   ./deploy/setup_server.sh

set -e

echo "========================================="
echo "  AI Trading Bot — Server Setup"
echo "========================================="

# --- Step 1: System updates ---
echo "[1/6] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# --- Step 2: Install Python 3.12+ ---
echo "[2/6] Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git

# --- Step 3: Create virtual environment ---
echo "[3/6] Creating virtual environment..."
cd /home/ubuntu/ai-trading-bot
python3 -m venv .venv
source .venv/bin/activate

# --- Step 4: Install dependencies ---
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install ccxt pandas numpy ta python-telegram-bot websockets aiosqlite pyyaml python-dotenv

# --- Step 5: Create .env if it doesn't exist ---
echo "[5/6] Checking .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env with your API keys!"
    echo "   nano .env"
    echo ""
fi

# --- Step 6: Install systemd service ---
echo "[6/6] Installing systemd service..."
sudo cp deploy/ai-trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-bot
echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "  Next steps:"
echo "  1. Edit your API keys:  nano .env"
echo "  2. Start the bot:       sudo systemctl start ai-trading-bot"
echo "  3. Check status:        sudo systemctl status ai-trading-bot"
echo "  4. View logs:           journalctl -u ai-trading-bot -f"
echo ""
echo "  The bot will auto-start on every reboot."
echo "========================================="

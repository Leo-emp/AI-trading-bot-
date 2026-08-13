# Deploy AI Trading Bot to Oracle Cloud (Free Forever)

## Step 1: Create Oracle Cloud Account (5 min)

1. Go to https://cloud.oracle.com
2. Click "Sign Up for Free"
3. Enter email, name, country
4. Add credit card (for verification only — you will NOT be charged)
5. Select your home region (pick closest to Binance servers: Singapore, Tokyo, or Frankfurt)

## Step 2: Create a Free VM (3 min)

1. Go to Oracle Cloud Console → Compute → Instances → Create Instance
2. Configure:
   - **Name:** ai-trading-bot
   - **Image:** Ubuntu 22.04 (or latest)
   - **Shape:** Click "Change Shape" → Ampere (ARM) → VM.Standard.A1.Flex
     - OCPUs: 1 (free tier allows up to 4)
     - Memory: 1 GB (free tier allows up to 24 GB)
   - **Networking:** Use defaults (auto-create VCN)
   - **SSH Key:** Click "Generate key pair" → Download BOTH keys (save them!)
3. Click "Create"
4. Wait 1-2 minutes for the VM to start
5. Copy the **Public IP Address** shown on the instance page

## Step 3: Connect to Your VM (1 min)

Open a terminal on your laptop:

```bash
# Windows (PowerShell):
ssh -i C:\path\to\your\ssh-key.key ubuntu@YOUR_IP_ADDRESS

# If permission error on Windows:
icacls C:\path\to\your\ssh-key.key /inheritance:r /grant:r "%USERNAME%:R"
```

## Step 4: Clone and Setup (5 min)

Run these commands on the VM:

```bash
# Clone your repo (replace with your actual GitHub URL)
cd /home/ubuntu
git clone https://github.com/Leo-emp/ai-trading-bot.git
cd ai-trading-bot

# Run the setup script
chmod +x deploy/setup_server.sh
./deploy/setup_server.sh
```

## Step 5: Add Your API Keys (2 min)

```bash
nano .env
```

Fill in:
```
BINANCE_API_KEY=your_key_here
BINANCE_SECRET=your_secret_here
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TRADING_MODE=paper
```

Save: Ctrl+O, Enter, Ctrl+X

## Step 6: Start the Bot (10 sec)

```bash
sudo systemctl start ai-trading-bot
```

That's it. The bot is now running 24/7.

## Useful Commands

| Command | What it does |
|---------|--------------|
| `sudo systemctl status ai-trading-bot` | Check if bot is running |
| `sudo systemctl stop ai-trading-bot` | Stop the bot |
| `sudo systemctl restart ai-trading-bot` | Restart the bot |
| `journalctl -u ai-trading-bot -f` | Watch live logs |
| `journalctl -u ai-trading-bot --since today` | Today's logs |
| `cat trading.log` | Full log file |

## The Bot Survives Everything

- **Server reboot:** systemd auto-starts it
- **Bot crash:** watchdog auto-restarts with backoff
- **Network drop:** WebSocket auto-reconnects with backoff
- **Your laptop off:** Doesn't matter — it's on the cloud

## When Ready for Live Trading

1. SSH into the server
2. Edit .env: `nano .env` → change `TRADING_MODE=live`
3. Restart: `sudo systemctl restart ai-trading-bot`

## Monthly Cost

$0. Oracle Cloud Free Tier is free forever (not a trial).

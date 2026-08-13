@echo off
:: start_bot.bat
:: Starts the AI trading bot via the watchdog (auto-restart on crash).
:: This script is called by Windows Task Scheduler on boot.
:: Logs go to trading.log in the project directory.

cd /d C:\Users\User\ai-trading-bot
C:\Python314\python.exe watchdog.py

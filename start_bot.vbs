' start_bot.vbs
' Launches the trading bot silently in the background (no CMD window).
' Place a shortcut to this file in your Startup folder.

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d C:\Users\User\ai-trading-bot && C:\Python314\python.exe watchdog.py", 0, False

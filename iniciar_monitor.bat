@echo off
title Monitor de alertas - Analizador Cripto
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -q -r requirements.txt
python monitor.py
pause

@echo off
title Analizador Cripto
cd /d "%~dp0"
if not exist .venv (
  echo Preparando por primera vez, espera unos minutos...
  python -m venv .venv || (echo No se encontro Python. Instalalo desde python.org marcando "Add Python to PATH". & pause & exit /b)
)
call .venv\Scripts\activate
pip install -q -r requirements.txt
streamlit run app.py
pause

@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run src\main.py
pause
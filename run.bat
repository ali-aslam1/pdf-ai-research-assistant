@echo off
REM Run Streamlit with suppressed transformers warnings
powershell -Command "$env:PYTHONWARNINGS='ignore'; streamlit run app.py 2>$null"

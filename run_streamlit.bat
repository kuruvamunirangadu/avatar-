
@echo off
REM Activate virtual environment
call ..\.venv310\Scripts\activate
REM Ensure Streamlit is installed
python -m pip show streamlit >nul 2>nul
if errorlevel 1 (
	echo Installing Streamlit...
	python -m pip install streamlit
)
REM Set PYTHONPATH to project root
set PYTHONPATH=%CD%
cd /d %~dp0
REM Run Streamlit app
streamlit run src/app/app.py

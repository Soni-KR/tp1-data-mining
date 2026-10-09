@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The project's Python environment was not found.
    echo Open README.md for setup instructions.
    pause
    exit /b 1
)
start "EduGuard API" ".venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000
start "EduGuard Dashboard" ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
echo EduGuard is starting. Keep the two new windows open.
echo Dashboard: http://localhost:8501
echo API documentation: http://localhost:8000/docs
echo In the dashboard, click Check API readiness before predicting.
echo If a program says its port is already in use, close its old server window first.
pause

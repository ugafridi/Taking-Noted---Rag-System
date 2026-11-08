@echo off
REM One-click launcher for the Course Notes RAG app (Windows).
cd /d "%~dp0backend"
start "RAG Backend :8000" cmd /k "call .venv\Scripts\activate.bat && uvicorn app.main:app --port 8000"
cd /d "%~dp0frontend"
start "RAG Frontend :5173" cmd /k "npm run dev"
echo.
echo Backend  -> http://localhost:8000/docs
echo Frontend -> http://localhost:5173
echo.

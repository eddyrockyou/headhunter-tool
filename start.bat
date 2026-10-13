@echo off
chcp 65001 >nul
cd /d %~dp0
where python >nul 2>nul
if errorlevel 1 (
  echo 找不到 Python。請先到 https://www.python.org/downloads/ 下載安裝，
  echo 安裝第一個畫面務必勾選「Add python.exe to PATH」，裝好後再雙擊本檔案。
  pause
  exit /b
)
if not exist .venv (
  echo 第一次啟動，正在安裝需要的套件，約需 1-3 分鐘…
  python -m venv .venv
  .venv\Scripts\pip install -q -r requirements.txt
)
echo 工具啟動中，瀏覽器會自動打開。使用期間請不要關閉這個視窗。
.venv\Scripts\python -m screener.app
pause

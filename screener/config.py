"""設定檔讀寫。設定存放在專案根目錄的 config.json。"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
CACHE_DIR = ROOT / ".cache"

DEFAULTS = {
    # 存放履歷的本機資料夾（會遞迴掃描子資料夾）
    "resume_folders": [],
    # Google Sheet 人才庫：[{"name": "人才庫", "url": "https://docs.google.com/spreadsheets/d/..."}]
    "google_sheets": [],
    # 私人 Sheet 才需要：Google 服務帳戶金鑰 JSON 路徑（需 pip install gspread）
    "google_service_account_file": "",
    "model": "claude-opus-5-5",
    # 評估深度：low / medium / high / xhigh / max（越高越準、越慢、越貴）
    "effort": "high",
    # 先用關鍵字粗篩後，交給 AI 深度評估的人數上限
    "shortlist_size": 20,
    # 同時評估的人數
    "max_workers": 4,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    if os.environ.get("RESUME_FOLDERS"):
        cfg["resume_folders"] = [p for p in os.environ["RESUME_FOLDERS"].split(os.pathsep) if p]
    return cfg


def save_config(cfg: dict) -> None:
    merged = dict(DEFAULTS)
    merged.update(cfg)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

"""從 Google Sheet 讀取人才庫，每一列 = 一位人選。

兩種讀法：
1. 公開連結（「知道連結的任何人都能檢視」）：直接下載 CSV，不需任何金鑰。
2. 私人 Sheet：設定 google_service_account_file，並把 Sheet 共用給服務帳戶的 email。
"""
import csv
import io
import re

import requests

NAME_HEADERS = ("姓名", "名字", "中文名", "英文名", "name", "candidate", "人選")


def parse_sheet_url(url: str) -> tuple[str, str | None]:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError(f"看不懂的 Google Sheet 網址：{url}")
    gid = re.search(r"[#&?]gid=(\d+)", url)
    return m.group(1), gid.group(1) if gid else None


def fetch_rows(url: str, service_account_file: str = "") -> list[list[str]]:
    sheet_id, gid = parse_sheet_url(url)
    if service_account_file:
        return _fetch_with_service_account(sheet_id, gid, service_account_file)

    export = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid:
        export += f"&gid={gid}"
    resp = requests.get(export, timeout=30)
    if resp.status_code != 200 or "text/html" in resp.headers.get("content-type", ""):
        raise PermissionError(
            "無法讀取 Google Sheet。請把共用設定改為「知道連結的任何人都能檢視」，"
            "或在設定中填入服務帳戶金鑰檔。"
        )
    resp.encoding = "utf-8"
    return list(csv.reader(io.StringIO(resp.text)))


def _fetch_with_service_account(sheet_id: str, gid: str | None, key_file: str) -> list[list[str]]:
    try:
        import gspread
    except ImportError as e:
        raise RuntimeError("讀取私人 Sheet 需要先執行：pip install gspread") from e
    gc = gspread.service_account(filename=key_file)
    sh = gc.open_by_key(sheet_id)
    ws = sh.get_worksheet_by_id(int(gid)) if gid else sh.sheet1
    return ws.get_all_values()


def rows_to_candidates(rows: list[list[str]], sheet_name: str, url: str) -> list[dict]:
    """把表格列轉成人選資料；自動找表頭列（第一個非空白列）。"""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return []
    headers = [h.strip() or f"欄位{i + 1}" for i, h in enumerate(rows[0])]
    name_idx = next(
        (i for i, h in enumerate(headers) if any(k in h.lower() for k in NAME_HEADERS)), None
    )
    candidates = []
    for row_no, row in enumerate(rows[1:], start=2):
        pairs = [(h, row[i].strip()) for i, h in enumerate(headers) if i < len(row) and row[i].strip()]
        if not pairs:
            continue
        name = row[name_idx].strip() if name_idx is not None and name_idx < len(row) else ""
        candidates.append({
            "id": f"sheet:{url}#{row_no}",
            "name": name or f"{sheet_name} 第 {row_no} 列",
            "source": f"Google Sheet「{sheet_name}」第 {row_no} 列",
            "source_type": "sheet",
            "text": "\n".join(f"{h}：{v}" for h, v in pairs),
        })
    return candidates

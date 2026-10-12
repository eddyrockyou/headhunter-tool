"""人才庫：掃描本機資料夾 + 讀取 Google Sheet，並快取已解析的文字。"""
import hashlib
import json
import re
import threading
from pathlib import Path

from .config import CACHE_DIR
from .extractors import MIN_PDF_CHARS, MIN_TEXT_CHARS, SUPPORTED_EXTS, ExtractionError, extract_text
from .sheets import fetch_rows, rows_to_candidates

FILE_CACHE = CACHE_DIR / "files.json"
_lock = threading.Lock()

# 從檔名猜姓名時要去掉的字
_NAME_NOISE = re.compile(
    r"(履歷表?|簡歷|自傳|作品集|resume|cv|curriculum vitae|104|1111|cakeresume|linkedin|"
    r"中文|英文|final|最新|\d{4,8}|[_\-\s()（）\[\]【】.]+)",
    re.IGNORECASE,
)


def guess_name(filename: str) -> str:
    name = _NAME_NOISE.sub(" ", Path(filename).stem).strip()
    return re.sub(r"\s+", " ", name) or Path(filename).stem


def _load_cache() -> dict:
    if FILE_CACHE.exists():
        try:
            return json.loads(FILE_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    FILE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def scan_folders(folders: list[str]) -> tuple[list[dict], list[str]]:
    """回傳 (人選清單, 警告訊息)。只重新解析新增或修改過的檔案。"""
    warnings = []
    candidates = []
    with _lock:
        cache = _load_cache()
        seen = set()
        texts = {}  # 內容相同的檔案（例如同一份履歷的 Word 與 PDF 版）只保留一份
        for folder in folders:
            root = Path(folder).expanduser()
            if not root.is_dir():
                warnings.append(f"找不到資料夾：{folder}")
                continue
            for path in sorted(root.rglob("*")):
                if path.suffix.lower() not in SUPPORTED_EXTS or path.name.startswith(("~$", ".")):
                    continue
                key = str(path.resolve())
                seen.add(key)
                stat = path.stat()
                sig = f"{stat.st_mtime_ns}:{stat.st_size}"
                entry = cache.get(key)
                if not entry or entry.get("sig") != sig:
                    try:
                        text = extract_text(path)
                        entry = {"sig": sig, "text": text, "error": ""}
                    except (ExtractionError, Exception) as e:  # 單一壞檔不影響其他檔案
                        entry = {"sig": sig, "text": "", "error": str(e)}
                    cache[key] = entry
                if entry["error"]:
                    warnings.append(f"{path.name}：{entry['error']}")
                    continue
                text = entry["text"].strip()
                is_pdf = path.suffix.lower() == ".pdf"
                scanned = is_pdf and len(text) < MIN_PDF_CHARS
                if not is_pdf and len(text) < MIN_TEXT_CHARS:
                    warnings.append(f"{path.name}：幾乎沒有文字內容，已略過")
                    continue
                if not scanned:
                    fingerprint = hashlib.sha1(re.sub(r"\s+", "", text).encode()).hexdigest()
                    if fingerprint in texts:
                        warnings.append(f"{path.name}：與 {texts[fingerprint]} 內容相同，已略過")
                        continue
                    texts[fingerprint] = path.name
                candidates.append({
                    "id": "file:" + hashlib.sha1(key.encode()).hexdigest()[:12],
                    "name": guess_name(path.name),
                    "source": str(path),
                    "source_type": "file",
                    "text": text,
                    # 掃描檔 PDF：評估時直接把 PDF 交給 AI 看
                    "pdf_path": str(path) if scanned else "",
                })
        for stale in set(cache) - seen:
            if not Path(stale).exists():
                del cache[stale]
        _save_cache(cache)
    return candidates, warnings


def load_sheets(sheets: list[dict], service_account_file: str = "") -> tuple[list[dict], list[str]]:
    candidates, warnings = [], []
    for s in sheets:
        url = s.get("url", "").strip()
        if not url:
            continue
        name = s.get("name") or "Google Sheet"
        try:
            rows = fetch_rows(url, service_account_file)
            candidates.extend(rows_to_candidates(rows, name, url))
        except Exception as e:
            warnings.append(f"{name}：{e}")
    return candidates, warnings


def load_library(cfg: dict) -> tuple[list[dict], list[str]]:
    files, w1 = scan_folders(cfg.get("resume_folders", []))
    rows, w2 = load_sheets(cfg.get("google_sheets", []), cfg.get("google_service_account_file", ""))
    return files + rows, w1 + w2

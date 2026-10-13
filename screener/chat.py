"""AI 助理對話：可以針對某次篩選結果提問，也可以直接搜尋整個人才庫。

每次提問時，先用問題在人才庫中找出最相關的幾位人選，連同篩選結果一起交給 AI 回答。
"""
import json
import re
import threading
import time

import anthropic

from .evaluator import SYSTEM_BASE
from .library import load_library
from .local_ai import Ollama
from .prefilter import rank
from .semantic import similarities

# 每次提問帶入的人選數與每份履歷字數（本機模型上下文較小）
LIMITS = {"ollama": (5, 2000), "anthropic": (10, 8000)}
# 人才庫在記憶體中保留的時間，避免每句話都重新讀 Google Sheet
LIBRARY_TTL = 300
MAX_HISTORY = 12

_lib = {"at": 0.0, "key": "", "items": []}
_lib_lock = threading.Lock()

CHAT_SYSTEM = SYSTEM_BASE + """
你現在是獵頭同事的 AI 助理，用對話方式協助他：比較人選、解釋評分、找人、寫邀約訊息或推薦信、
規劃面試問題、分析職缺等。回答要直接、具體、精簡；需要條列時用「- 」開頭。
每次提問時系統會附上從人才庫檢索到的人選資料（不一定都相關），以及使用者選定的篩選結果。
只根據這些資料回答人選相關的事實；資料裡沒有的就說沒有，不要編造人選或經歷。
提到人選時寫出姓名與來源檔名，方便同事找到原始履歷。"""


def _library(cfg: dict) -> list[dict]:
    key = json.dumps([cfg.get("resume_folders"), cfg.get("google_sheets")], ensure_ascii=False)
    with _lib_lock:
        if _lib["key"] != key or time.time() - _lib["at"] > LIBRARY_TTL:
            items, _ = load_library(cfg)
            _lib.update(at=time.time(), key=key, items=[c for c in items if c["text"].strip() and not c.get("pdf_path")])
        return _lib["items"]


def search_library(cfg: dict, query: str, limit: int) -> list[dict]:
    library = _library(cfg)
    if not library:
        return []
    semantic = None
    if cfg.get("embedding_model"):
        try:
            semantic = similarities(Ollama(cfg["ollama_url"]), cfg["embedding_model"], query, library)
        except Exception:
            semantic = None  # 語意模型不可用時只用文字比對
    # 問題中的英文詞（技能、證照、公司名，如 Kubernetes、PMP、TSMC）當作關鍵字精準比對
    terms = {t for t in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}", query)}
    keywords = [{"term": t, "synonyms": [], "must_have": True} for t in terms]
    return rank(library, query, keywords, limit, semantic)


def _run_context(run: dict) -> str:
    a = run.get("analysis") or {}
    lines = [f"職缺：{a.get('job_title', '')}｜{a.get('company', '')}",
             "JD：" + run["jd"][:3000],
             "補充資訊：" + (run.get("extra") or "（無）"),
             "必要條件：" + "；".join(a.get("must_haves", [])),
             "", "評估結果（依分數排序）："]
    for i, r in enumerate(run["results"][:15], 1):
        e = r["evaluation"]
        lines.append(json.dumps({
            "排名": i, "姓名": e["candidate_name"], "分數": e["score"], "合適度": e["fit_level"],
            "現職": f"{e['current_title']} @ {e['current_company']}", "年資": e["years_experience"],
            "總結": e["one_line_summary"], "推薦原因": e["recommendation_reasons"], "落差": e["gaps"],
            "風險": [f"[{x['severity']}] {x['risk']}" for x in e["risks"]],
            "待確認": [x["question"] for x in e["info_to_clarify"]], "來源": r["source"],
        }, ensure_ascii=False))
    if run.get("summary"):
        lines += ["", "推薦總結：" + run["summary"]["overview"]]
    return "\n".join(lines)


def build_prompt(cfg: dict, message: str, run: dict | None, search: bool) -> tuple[str, list[dict]]:
    """回傳 (這一輪要送給 AI 的完整訊息, 檢索到的人選清單)。"""
    engine = cfg.get("engine", "ollama")
    n, chars = LIMITS.get(engine, LIMITS["ollama"])
    parts, found = [], []
    if run:
        parts.append(f"<screening_result>\n{_run_context(run)}\n</screening_result>")
    if search:
        found = search_library(cfg, message, n)
        if found:
            parts.append("<talent_pool_matches>\n" + "\n\n".join(
                f"### {c['name']}（來源：{c['source']}）\n{c['text'][:chars]}" for c in found
            ) + "\n</talent_pool_matches>")
    parts.append(f"<question>\n{message}\n</question>")
    return "\n\n".join(parts), found


def stream_reply(cfg: dict, history: list[dict], prompt: str):
    """逐段產生 AI 回覆。history 為先前的 [{role, content}]，只含對話文字。"""
    history = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
    history = history[-MAX_HISTORY:]
    while history and history[0]["role"] != "user":
        history.pop(0)
    messages = history + [{"role": "user", "content": prompt}]

    if cfg.get("engine", "ollama") == "ollama":
        ollama = Ollama(cfg["ollama_url"])
        yield from ollama.chat_stream(cfg.get("local_model") or "qwen3:8b",
                                      [{"role": "system", "content": CHAT_SYSTEM}] + messages)
        return

    model = cfg.get("model") or "claude-opus-5-5"
    kwargs = {}
    if "haiku" not in model:
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    with anthropic.Anthropic().beta.messages.stream(
        model=model, max_tokens=16000, system=CHAT_SYSTEM, messages=messages,
        output_config={"effort": "medium"}, **kwargs,
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n（AI 拒絕回答這個問題，請換個方式描述）"

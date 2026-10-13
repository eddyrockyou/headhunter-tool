"""語意比對：用本機 embedding 模型把履歷轉成向量，找出「意思相近」的人選。

關鍵字比對會漏掉寫法不同的履歷（例如 JD 寫「業務開發」、履歷寫「BD」或「拓展新客戶」），
語意比對可以補上。向量會快取，每份履歷只需計算一次。
"""
import hashlib
import json
import math
import re

from .config import CACHE_DIR
from .local_ai import Ollama

# 每份履歷取前面這麼多字計算向量（職稱、經歷摘要通常在前面）
EMBED_CHARS = 4000
BATCH = 16


def _cache_path(model: str):
    return CACHE_DIR / "embeddings" / (re.sub(r"[^\w.-]", "_", model) + ".json")


def _key(text: str) -> str:
    return hashlib.sha1(text[:EMBED_CHARS].encode()).hexdigest()


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [round(x / n, 5) for x in v]


def similarities(ollama: Ollama, model: str, query: str, candidates: list[dict],
                 progress=None) -> dict[str, float]:
    """回傳 {人選 id: 與 query 的相似度 0–1}。"""
    path = _cache_path(model)
    cache = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    with_text = [c for c in candidates if c["text"].strip()]
    todo = list({_key(c["text"]): c["text"][:EMBED_CHARS] for c in with_text
                 if _key(c["text"]) not in cache}.items())
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        vecs = ollama.embed(model, [t for _, t in chunk])
        for (k, _), v in zip(chunk, vecs):
            cache[k] = _normalize(v)
        if progress:
            progress(min(i + BATCH, len(todo)), len(todo))
        if (i // BATCH) % 20 == 19:  # 定期存檔，中途關掉也不會白算
            _save(path, cache)
    if todo:
        _save(path, cache)

    q = _normalize(ollama.embed(model, [query[:EMBED_CHARS]])[0])
    out = {}
    for c in with_text:
        v = cache[_key(c["text"])]
        out[c["id"]] = max(0.0, sum(a * b for a, b in zip(q, v)))
    return out


def _save(path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")

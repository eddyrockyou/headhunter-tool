"""本機粗篩：快速排序整個人才庫，把最相關的人選交給 AI 深度評估。

人才庫可能有上千份履歷，全部交給 AI 會太慢，所以先用
關鍵字（含同義詞）、語意相似度（若有開啟）、中英文字詞重疊度做初步排序。
"""
import math
import re
from collections import Counter

_WORD = re.compile(r"[a-z][a-z0-9+#.\-]{1,}|[一-鿿]")


def _tokens(text: str) -> Counter:
    """英文取單字，中文取相鄰兩字（bigram）。"""
    text = text.lower()
    toks = _WORD.findall(text)
    out = Counter()
    prev_cjk = None
    for t in toks:
        if len(t) == 1 and "一" <= t <= "鿿":
            if prev_cjk:
                out[prev_cjk + t] += 1
            prev_cjk = t
        else:
            out[t] += 1
            prev_cjk = None
    return out


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def keyword_score(text: str, keywords: list[dict]) -> tuple[float, list[str]]:
    low = text.lower()
    score, hits = 0.0, []
    for kw in keywords:
        terms = [kw["term"], *kw.get("synonyms", [])]
        found = [t for t in terms if t and t.lower() in low]
        if found:
            weight = 3.0 if kw.get("must_have") else 1.0
            count = sum(low.count(t.lower()) for t in found)
            score += weight * (1 + math.log(count))
            hits.append(kw["term"])
    return score, hits


def rank(candidates: list[dict], jd_text: str, keywords: list[dict], limit: int,
         semantic: dict[str, float] | None = None) -> list[dict]:
    """回傳排序後要深度評估的人選（每人加上 prefilter_score / keyword_hits）。

    semantic：{人選 id: 語意相似度}，有提供時納入排序。
    掃描檔 PDF 沒有文字可比對，一律保留給 AI 直接看原檔。
    """
    jd_vec = _tokens(jd_text)
    if semantic:
        lo, hi = min(semantic.values()), max(semantic.values())
        span = (hi - lo) or 1.0
    max_kw = sum(3.0 if k.get("must_have") else 1.0 for k in keywords) or 1.0
    scored, scanned = [], []
    for c in candidates:
        if c.get("pdf_path"):
            scanned.append({**c, "prefilter_score": None, "keyword_hits": []})
            continue
        kw, hits = keyword_score(c["text"], keywords)
        sim = _cosine(jd_vec, _tokens(c["text"]))
        kw_part = min(kw / max_kw, 1.0)
        sim_part = min(sim * 3, 1.0)
        if semantic:
            sem_part = (semantic.get(c["id"], lo) - lo) / span
            total = 0.45 * kw_part + 0.45 * sem_part + 0.10 * sim_part
        else:
            total = 0.75 * kw_part + 0.25 * sim_part
        scored.append({**c, "prefilter_score": round(total * 100, 1), "keyword_hits": hits})
    scored.sort(key=lambda c: c["prefilter_score"], reverse=True)
    return scored[: max(limit - len(scanned), 0)] + scanned

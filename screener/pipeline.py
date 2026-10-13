"""完整篩選流程：分析 JD → 粗篩 → AI 逐一評估 → 總結推薦。"""
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import CACHE_DIR
from .evaluator import Evaluator
from .library import load_library
from .local_ai import Ollama
from .prefilter import rank
from .semantic import similarities

RUNS_DIR = CACHE_DIR / "runs"


class Run:
    """一次篩選的狀態，網頁會定期讀取以顯示進度。"""

    def __init__(self, jd: str, extra: str, cfg: dict):
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.jd, self.extra, self.cfg = jd, extra, cfg
        self.status = "running"
        self.stage = "準備中"
        self.progress = [0, 0]
        self.warnings: list[str] = []
        self.library_size = 0
        self.analysis: dict | None = None
        self.results: list[dict] = []
        self.failed: list[dict] = []
        self.summary: dict | None = None
        self.texts: dict[str, str] = {}  # 入選者的履歷全文，供「匯出給 Claude」使用
        self.error = ""
        self.created_at = time.strftime("%Y-%m-%d %H:%M")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "warnings": self.warnings,
            "library_size": self.library_size,
            "jd": self.jd,
            "extra": self.extra,
            "analysis": self.analysis,
            "results": sorted(self.results, key=lambda r: r["evaluation"]["score"], reverse=True),
            "failed": self.failed,
            "summary": self.summary,
            "error": self.error,
            "created_at": self.created_at,
        }

    def save(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / f"{self.id}.json").write_text(
            json.dumps({**self.to_dict(), "texts": self.texts}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def execute(self) -> None:
        try:
            self._execute()
            self.status = "done"
            self.stage = "完成"
        except Exception as e:
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.save()

    def _execute(self) -> None:
        cfg = self.cfg
        ev = Evaluator(cfg)
        local = cfg.get("engine", "ollama") == "ollama"
        ollama = Ollama(cfg.get("ollama_url") or "http://127.0.0.1:11434")
        if local and not ollama.status()["running"]:
            raise RuntimeError("Ollama 沒有在執行。請先開啟 Ollama（開始選單搜尋 Ollama），再重新按「開始推薦人選」")

        self.stage = "讀取人才庫（本機資料夾、Google Sheet）"
        library, warnings = load_library(cfg, progress=self._progress("解析履歷檔案"))
        self.warnings.extend(warnings)
        if local:
            scanned = [c for c in library if c.get("pdf_path")]
            if scanned:
                self.warnings.append(f"{len(scanned)} 份掃描檔（圖片型 PDF）本機 AI 無法閱讀，已略過："
                                     + "、".join(c["name"] for c in scanned[:20])
                                     + ("…" if len(scanned) > 20 else ""))
                library = [c for c in library if not c.get("pdf_path")]
        self.library_size = len(library)
        if not library:
            raise RuntimeError("人才庫是空的，請先在「設定」加入履歷資料夾或 Google Sheet")
        self.progress = [0, 0]

        self.stage = "AI 分析職缺需求"
        self.analysis = ev.analyze_jd(self.jd, self.extra)
        query = self.jd + "\n" + self.extra

        semantic = None
        if cfg.get("embedding_model"):
            self.stage = "語意比對整個人才庫"
            try:
                semantic = similarities(ollama, cfg["embedding_model"], query, library,
                                        progress=self._progress("建立語意索引（每份履歷只需算一次）"))
            except Exception as e:
                self.warnings.append(f"語意比對未啟用，只用關鍵字粗篩：{e}")
            self.progress = [0, 0]

        self.stage = "粗篩排序"
        shortlist = rank(library, query, self.analysis["keywords"],
                         int(cfg["shortlist_size"]), semantic)
        self.progress = [0, len(shortlist)]

        self.stage = f"AI 深度評估 {len(shortlist)} 位人選"
        with ThreadPoolExecutor(max_workers=int(cfg["max_workers"])) as pool:
            futures = {
                pool.submit(ev.evaluate_candidate, self.jd, self.extra, self.analysis, c): c
                for c in shortlist
            }
            for fut in as_completed(futures):
                c = futures[fut]
                self.texts[c["id"]] = c["text"]
                try:
                    self.results.append({
                        "id": c["id"],
                        "name": c["name"],
                        "source": c["source"],
                        "source_type": c["source_type"],
                        "prefilter_score": c["prefilter_score"],
                        "keyword_hits": c["keyword_hits"],
                        "evaluation": fut.result(),
                    })
                except Exception as e:
                    self.failed.append({"name": c["name"], "source": c["source"], "error": str(e)})
                self.progress = [self.progress[0] + 1, len(shortlist)]

        if self.results:
            self.stage = "產出推薦總結"
            self.summary = ev.summarize(self.analysis, self.to_dict()["results"])


    def _progress(self, label: str):
        def update(done: int, total: int) -> None:
            self.stage = label
            self.progress = [done, total]
        return update


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for p in sorted(RUNS_DIR.glob("*.json"), reverse=True)[:50]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        title = (d.get("analysis") or {}).get("job_title") or d["jd"][:30]
        out.append({"id": d["id"], "title": title, "created_at": d["created_at"],
                    "status": d["status"], "count": len(d["results"])})
    return out


def load_run(run_id: str) -> dict | None:
    p = RUNS_DIR / f"{run_id}.json"
    if p.parent != RUNS_DIR or not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

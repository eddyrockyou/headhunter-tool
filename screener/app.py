"""本機網頁介面。執行：python -m screener.app ，瀏覽器開 http://127.0.0.1:5000"""
import csv
import io
import os
import re
import threading
import webbrowser

from flask import Flask, Response, abort, jsonify, render_template, request

from .config import load_config, save_config
from .library import load_library
from .pipeline import Run, list_runs, load_run

app = Flask(__name__)
app.json.ensure_ascii = False
_runs: dict[str, Run] = {}
_RUN_ID = re.compile(r"^[\w-]+$")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/config")
def get_config():
    cfg = load_config()
    cfg["api_key_set"] = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    return jsonify(cfg)


@app.post("/api/config")
def set_config():
    data = request.get_json()
    cfg = load_config()
    for key in ("resume_folders", "google_sheets", "google_service_account_file",
                "model", "effort", "shortlist_size", "max_workers"):
        if key in data:
            cfg[key] = data[key]
    save_config(cfg)
    return jsonify({"ok": True})


@app.get("/api/library")
def library():
    candidates, warnings = load_library(load_config())
    return jsonify({
        "count": len(candidates),
        "candidates": [
            {"name": c["name"], "source": c["source"], "source_type": c["source_type"],
             "chars": len(c["text"]), "scanned": bool(c.get("pdf_path"))}
            for c in candidates
        ],
        "warnings": warnings,
    })


@app.post("/api/runs")
def start_run():
    data = request.get_json()
    jd = (data.get("jd") or "").strip()
    if len(jd) < 20:
        return jsonify({"error": "請貼上完整的職缺 JD"}), 400
    run = Run(jd, (data.get("extra") or "").strip(), load_config())
    _runs[run.id] = run
    threading.Thread(target=run.execute, daemon=True).start()
    return jsonify({"id": run.id})


@app.get("/api/runs")
def runs():
    return jsonify(list_runs())


def _get_run(run_id: str) -> dict:
    if not _RUN_ID.match(run_id):
        abort(404)
    if run_id in _runs:
        return _runs[run_id].to_dict()
    data = load_run(run_id)
    if not data:
        abort(404)
    return data


@app.get("/api/runs/<run_id>")
def run_detail(run_id):
    return jsonify(_get_run(run_id))


@app.get("/api/runs/<run_id>/export.csv")
def export_csv(run_id):
    d = _get_run(run_id)
    buf = io.StringIO()
    buf.write("﻿")  # 讓 Excel 正確顯示中文
    w = csv.writer(buf)
    w.writerow(["排名", "姓名", "評分", "合適度", "現職", "公司", "年資", "一句話總結",
                "推薦原因", "落差", "風險", "需補充了解", "溝通切入點", "來源"])
    for i, r in enumerate(d["results"], 1):
        e = r["evaluation"]
        w.writerow([
            i, e["candidate_name"], e["score"], e["fit_level"], e["current_title"],
            e["current_company"], e["years_experience"], e["one_line_summary"],
            "\n".join(e["recommendation_reasons"]), "\n".join(e["gaps"]),
            "\n".join(f"[{x['severity']}] {x['risk']}：{x['detail']}" for x in e["risks"]),
            "\n".join(f"{x['question']}（{x['why']}）" for x in e["info_to_clarify"]),
            e["pitch_angle"], r["source"],
        ])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=screening-{run_id}.csv"})


@app.get("/api/runs/<run_id>/export.md")
def export_md(run_id):
    d = _get_run(run_id)
    a = d.get("analysis") or {}
    lines = [f"# 人選推薦報告：{a.get('job_title', '')}", "", f"產出時間：{d['created_at']}", ""]
    s = d.get("summary")
    if s:
        lines += ["## 推薦總結", "", s["overview"], ""]
        lines += [f"{i}. **{p['candidate_name']}** — {p['why_this_rank']}" for i, p in enumerate(s["top_picks"], 1)]
        lines += ["", "### 下一步", *[f"- {x}" for x in s["next_steps"]], ""]
    for i, r in enumerate(d["results"], 1):
        e = r["evaluation"]
        lines += [
            f"## {i}. {e['candidate_name']}｜{e['score']} 分｜{e['fit_level']}", "",
            f"{e['current_title']} @ {e['current_company']}（{e['years_experience']}）", "",
            f"> {e['one_line_summary']}", "",
            "**推薦原因**", *[f"- {x}" for x in e["recommendation_reasons"]], "",
            "**落差**", *[f"- {x}" for x in e["gaps"]], "",
            "**風險考量**", *[f"- [{x['severity']}] {x['risk']}：{x['detail']}（驗證方式：{x['how_to_verify']}）" for x in e["risks"]], "",
            "**需要補充了解**", *[f"- {x['question']}（{x['why']}）" for x in e["info_to_clarify"]], "",
            f"**溝通切入點**：{e['pitch_angle']}", "", f"來源：{r['source']}", "",
        ]
    return Response("\n".join(lines), mimetype="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=screening-{run_id}.md"})


def main():
    port = int(os.environ.get("PORT", 5000))
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    # 只綁定本機，履歷資料不會對外開放
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()

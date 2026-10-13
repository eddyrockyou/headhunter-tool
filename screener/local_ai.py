"""本機免費 AI：透過 Ollama（https://ollama.com）在自己電腦上執行開源模型。

履歷不會離開你的電腦，也不需要 API 金鑰。
"""
import json

import requests


class LocalAIError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str = "http://127.0.0.1:11434"):
        self.url = url.rstrip("/")

    def status(self) -> dict:
        """回傳 {"running": bool, "models": [已下載的模型名稱]}"""
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            r.raise_for_status()
            return {"running": True, "models": [m["name"] for m in r.json().get("models", [])]}
        except (requests.RequestException, ValueError):
            return {"running": False, "models": []}

    def _post(self, path: str, body: dict, timeout: int) -> dict:
        try:
            r = requests.post(f"{self.url}{path}", json=body, timeout=timeout)
        except requests.ConnectionError as e:
            raise LocalAIError("連不到 Ollama，請確認 Ollama 已開啟（工作列右下角有羊駝圖示）") from e
        except requests.Timeout as e:
            raise LocalAIError("本機 AI 回應逾時，電腦可能太忙；可改用較小的模型或減少同時評估人數") from e
        if r.status_code == 404:
            raise LocalAIError(f"找不到模型「{body.get('model')}」，請先在命令提示字元執行：ollama pull {body.get('model')}")
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", r.text)
            except ValueError:
                msg = r.text
            raise LocalAIError(f"Ollama 錯誤：{msg}")
        return r.json()

    def chat_json(self, model: str, system: str, user: str, schema: dict, num_ctx: int = 16384) -> dict:
        """要求模型依 JSON schema 回答並解析。"""
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            # 關閉「思考」模式：速度快很多，結構化評估的品質差異不大
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": num_ctx},
        }
        try:
            data = self._post("/api/chat", body, timeout=1800)
        except LocalAIError as e:
            if "think" not in str(e):
                raise
            body.pop("think")  # 不支援思考模式的模型
            data = self._post("/api/chat", body, timeout=1800)
        content = data.get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LocalAIError("本機 AI 回傳的格式不完整，請重試或換用較大的模型") from e

    def chat_stream(self, model: str, messages: list[dict], num_ctx: int = 16384):
        """逐段產生回覆文字（聊天用，讓文字邊想邊出現）。"""
        body = {"model": model, "messages": messages, "stream": True, "think": False,
                "options": {"temperature": 0.4, "num_ctx": num_ctx}}
        try:
            r = requests.post(f"{self.url}/api/chat", json=body, stream=True, timeout=1800)
            if r.status_code == 400 and "think" in r.text:
                body.pop("think")  # 不支援思考模式的模型
                r = requests.post(f"{self.url}/api/chat", json=body, stream=True, timeout=1800)
        except requests.ConnectionError as e:
            raise LocalAIError("連不到 Ollama，請確認 Ollama 已開啟（工作列右下角有羊駝圖示）") from e
        with r:
            if r.status_code == 404:
                raise LocalAIError(f"找不到模型「{model}」，請先在命令提示字元執行：ollama pull {model}")
            if r.status_code >= 400:
                raise LocalAIError(f"Ollama 錯誤：{r.text}")
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LocalAIError(f"Ollama 錯誤：{chunk['error']}")
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield text
                if chunk.get("done"):
                    break

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": model, "input": texts}, timeout=600)
        return data["embeddings"]

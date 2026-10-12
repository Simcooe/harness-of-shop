"""调用模型代理（OpenAI 兼容接口）并解析 JSON 输出，带磁盘缓存：同一模型、同一组消息只请求一次。"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "llm"


def _dotenv() -> dict:
    path = ROOT / ".env"
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_ENV = _dotenv()
BASE_URL = os.environ.get("MODEL_PROXY_BASE_URL", _ENV.get("MODEL_PROXY_BASE_URL", "")).rstrip("/")
API_KEY = os.environ.get("MODEL_PROXY_API_KEY", _ENV.get("MODEL_PROXY_API_KEY", ""))


def _parse_json(text: str):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _read_cache(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, record: dict) -> None:
    # 多个评测进程可能同时写同一个缓存文件：先写临时文件再原子替换
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _post(payload: dict, timeout: int) -> dict:
    req = Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def chat_text(model: str, messages: list, max_tokens: int = 200, timeout: int = 120, retries: int = 4) -> str:
    """temperature 0、关闭思考的普通对话（模拟买家用），结果按请求内容缓存。"""
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "enable_thinking": False}
    key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cache = CACHE_DIR / f"{key}.json"
    cached = _read_cache(cache)
    if cached is not None:
        return cached["content"]
    last_error = None
    for attempt in range(retries):
        try:
            body = _post(payload, timeout)
            content = (body["choices"][0]["message"]["content"] or "").strip()
            _write_cache(cache, {"model": model, "content": content, "usage": body.get("usage")})
            return content
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{model} 调用失败: {last_error}")


def chat_json(model: str, system: str, user: str, max_tokens: int = 4096, timeout: int = 300, retries: int = 4):
    """temperature 0、关闭思考，要求返回一个 JSON 对象；结果按请求内容缓存。"""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cache = CACHE_DIR / f"{key}.json"
    cached = _read_cache(cache)
    if cached is not None:
        return cached["parsed"]

    last_error = None
    for attempt in range(retries):
        try:
            body = _post(payload, timeout)
            content = body["choices"][0]["message"]["content"] or ""
            parsed = _parse_json(content)
            _write_cache(cache, {"model": model, "content": content, "parsed": parsed, "usage": body.get("usage")})
            return parsed
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{model} 调用失败: {last_error}")

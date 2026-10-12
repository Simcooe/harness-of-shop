"""把一条轨迹的日志（模型消息、环境步骤、插件事件、终局）重建成按回合排序的统一时间线。"""
import json
import re
from pathlib import Path


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if c.get("type") == "text")


class Trajectory:
    def __init__(self, task_dir: Path):
        self.dir = task_dir
        self.task_id = int(task_dir.name.split("_")[-1])
        reset_path = task_dir / "reset_public.json"
        self.reset = json.loads(reset_path.read_text(encoding="utf-8")) if reset_path.exists() else {}
        self.query = self.reset.get("query", "")
        # 多轮实验里 agent 拿到的是模糊需求；单轮时两者相同
        self.agent_query = self.reset.get("agent_query") or self.query
        self.initial_state = self.reset.get("observation_state") or {}
        self.steps = load_jsonl(task_dir / "env_steps.jsonl")
        self.harness = load_jsonl(task_dir / "harness_events.jsonl")
        self.asks = [r for r in load_jsonl(task_dir / "ask_log.jsonl") if r.get("type", "ask") == "ask"]
        terminal_path = task_dir / "terminal.json"
        self.terminal = json.loads(terminal_path.read_text(encoding="utf-8"))["result"] if terminal_path.exists() else None
        done_path = task_dir / "done.json"
        self.done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}

        self.assistant = []  # 每个元素对应一个 LLM 回合（从 1 开始编号）
        for f in sorted((task_dir / "session").glob("*.jsonl")):
            for row in load_jsonl(f):
                msg = row.get("message") or {}
                if row.get("type") == "message" and msg.get("role") == "assistant":
                    calls = [c for c in msg.get("content") or [] if c.get("type") == "toolCall"]
                    self.assistant.append(
                        {
                            "turn": len(self.assistant) + 1,
                            "text": text_of(msg.get("content")).strip(),
                            "calls": [{"name": c.get("name"), "args": c.get("arguments")} for c in calls],
                            "usage": msg.get("usage") or {},
                            "stop": msg.get("stopReason"),
                        }
                    )

    @property
    def purchase(self) -> dict:
        return ((self.terminal or {}).get("purchase") or {}) if self.terminal else {}

    @property
    def purchased(self) -> bool:
        return bool(self.purchase.get("asin"))

    def state_before(self, index: int) -> dict:
        """第 index 个环境步骤（从 0 开始）执行前 agent 所在的页面。"""
        for i in range(index - 1, -1, -1):
            result = self.steps[i].get("result") or {}
            if not result.get("done"):
                return result.get("observation_state") or {}
        return self.initial_state

    def buy_index(self):
        for i, s in enumerate(self.steps):
            if (s.get("result") or {}).get("done") and s.get("tool") == "buy_now":
                return i
        return None

    def buy_turn(self):
        i = self.buy_index()
        return self.steps[i].get("turn") if i is not None else None

    @property
    def protocol(self) -> str:
        """single（单轮）、ours（我们自己的多轮设置）或 paper（论文的多轮设置）；旧运行没有记录时按 multiturn 推断。"""
        config = self.dir.parent / "run_config.json"
        if config.exists():
            proto = json.loads(config.read_text(encoding="utf-8")).get("protocol")
            if proto:
                return proto
        return "ours" if self.multiturn else "single"

    @property
    def multiturn(self) -> bool:
        """是否来自多轮实验（模糊需求）的运行；对照组的题需求完整，但也算。"""
        config = self.dir.parent / "run_config.json"
        if config.exists() and json.loads(config.read_text(encoding="utf-8")).get("multiturn"):
            return True
        return self.agent_query != self.query or bool(self.asks)

    @property
    def can_ask(self) -> bool:
        return (self.dir / "shopper.json").exists()

    def timeline(self) -> list:
        """按回合合并：先是该回合模型说的话，再是该回合的环境步骤、提问和插件事件。"""
        events = []
        for a in self.assistant:
            events.append({"kind": "assistant", "turn": a["turn"], "order": 0, **a})
        for i, s in enumerate(self.steps):
            events.append({"kind": "step", "turn": s.get("turn"), "order": 1, "index": i, "step": s.get("step"), **s})
        for j, q in enumerate(self.asks):
            events.append({"kind": "ask", "turn": q.get("turn"), "order": 1, "index": 10_000 + j, **q})
        for j, e in enumerate(self.harness):
            events.append({"kind": "harness", "turn": e.get("turn"), "order": 2, "seq": j, **e})
        return sorted(events, key=lambda e: (e.get("turn") or 0, e["order"], e.get("index", e.get("seq", 0))))


def visible_text_about(state: dict, asin: str) -> str:
    """一个页面上与指定商品相关、agent 能看到的全部文字。"""
    asin = str(asin)
    page = state.get("page_type")
    parts = []
    if page == "search_results":
        for p in state.get("products") or []:
            if str(p.get("asin")) == asin:
                parts += [p.get("title", ""), p.get("brand", ""), p.get("category", ""), str(p.get("price", "")), "；".join(p.get("key_attributes") or [])]
    elif page in ("product_detail", "information_subpage"):
        p = state.get("product") or {}
        if str(p.get("asin")) == asin:
            parts += [p.get("title", ""), p.get("brand", ""), p.get("category", ""), str(p.get("price", "")), "；".join(p.get("key_attributes") or [])]
            parts.append(json.dumps(state.get("available_options") or {}, ensure_ascii=False))
            parts.append(json.dumps(state.get("selected_options") or {}, ensure_ascii=False))
            if state.get("selected_price") is not None:
                parts.append(f"当前规格价格 {state.get('selected_price')}")
            content = state.get("content")
            if content:
                parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
    return "\n".join(str(x) for x in parts if x)


def squash(text) -> str:
    return re.sub(r"[\s，。,.！!？?、；;：:“”\"'（）()【】\[\]|/]", "", str(text)).lower()


def norm(text) -> str:
    # 环境渲染规格值时会把 "/" 显示成 " | "，比较前统一
    return re.sub(r"\s+", "", str(text)).lower().replace("|", "/")


def gold_value_matches(gold_value, value) -> bool:
    # 少数 gold 标注里的 emoji 被编码成了 "?"（如 "➕" → "?"），把它当作单字符通配
    pattern = "".join("." if ch == "?" else re.escape(ch) for ch in norm(gold_value))
    return re.fullmatch(pattern, norm(value), re.S) is not None


def options_match(selected: dict, gold_options: list) -> bool:
    values = list((selected or {}).values())
    return all(any(gold_value_matches(o, v) for v in values) for o in gold_options)

"""论文（ShopSimulator 官方 multi_eval）的多轮规则：买家的系统提示和开场请求。

照官方 shopper.reset 的拼法：官方配置里的买家提示 + 本题完整需求 + 标准答案的规格选项（官方让买家知道自己要哪个规格）。
提示原文来自官方仓库（没有许可证），所以放在 gitignore 的 third_party/ 下，不进项目目录。
与官方的差异：官方把"搜索功能是否可用/可点击的按钮"这段环境文字也误拼进了买家的任务描述，这里不拼。

画像设置（多轮 + 画像）：官方没有公开这个设置的评测代码，按论文的做法自己定：买家只知道简版需求（instruction_simple）
和规格选项，其余细节在用户画像里，画像只给 agent；买家提示末尾加一条我们自己的规则（prompts/shopper_persona.md）：
被问到不知道的细节就让 agent 帮着挑，不自己编。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOPPER_PROMPT = ROOT / "third_party" / "shopsim_official" / "shopper_paper.md"
PERSONA_SHOPPER_RULE = ROOT / "prompts" / "shopper_persona.md"
PERSONA_AGENT_BLOCK = ROOT / "prompts" / "persona_agent.md"
# 官方 Agent.reset 里先发给买家的那句话，买家的回复就是开场白
OPENING_REQUEST = "请提供您的购物需求。"


def shopper_system(full_query: str, goal_options, prompt_path: Path = SHOPPER_PROMPT) -> str:
    template = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    options = json.dumps(list(goal_options or []), ensure_ascii=False)
    return template.replace("{full_query}", full_query).replace("{goal_options}", options)


def short_query(reset: dict) -> str:
    text = str(reset.get("instruction_simple") or "").strip()
    if not text:
        raise RuntimeError("reset 没有返回简版需求")
    return text


def persona_profile(reset: dict) -> dict:
    profile = reset.get("user_persona")
    if not isinstance(profile, dict) or not profile:
        raise RuntimeError("这道题没有用户画像")
    return {k: v for k, v in profile.items() if k != "__reasoning__"}


def persona_shopper_system(short: str, goal_options, prompt_path: Path = SHOPPER_PROMPT) -> str:
    rule = PERSONA_SHOPPER_RULE.read_text(encoding="utf-8").strip()
    return f"{shopper_system(short, goal_options, prompt_path)}\n\n{rule}"


def persona_agent_block(profile: dict) -> str:
    template = PERSONA_AGENT_BLOCK.read_text(encoding="utf-8").strip()
    return template.replace("{persona}", json.dumps(profile, ensure_ascii=False))

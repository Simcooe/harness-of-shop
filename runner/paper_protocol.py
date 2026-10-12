"""论文（ShopSimulator 官方 multi_eval）的多轮规则：买家的系统提示和开场请求。

照官方 shopper.reset 的拼法：官方配置里的买家提示 + 本题完整需求 + 标准答案的规格选项（官方让买家知道自己要哪个规格）。
提示原文来自官方仓库（没有许可证），所以放在 gitignore 的 third_party/ 下，不进项目目录。
与官方的差异：官方把"搜索功能是否可用/可点击的按钮"这段环境文字也误拼进了买家的任务描述，这里不拼。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOPPER_PROMPT = ROOT / "third_party" / "shopsim_official" / "shopper_paper.md"
# 官方 Agent.reset 里先发给买家的那句话，买家的回复就是开场白
OPENING_REQUEST = "请提供您的购物需求。"


def shopper_system(full_query: str, goal_options, prompt_path: Path = SHOPPER_PROMPT) -> str:
    template = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    options = json.dumps(list(goal_options or []), ensure_ascii=False)
    return template.replace("{full_query}", full_query).replace("{goal_options}", options)

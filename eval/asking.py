"""多轮提问评测：每个问题问到了哪些约束、模拟买家答得对不对、失败是“该问没问”还是“问了没用”。

约束分三类：hidden（从需求里删掉、只能问出来的规格约束）、known（agent 拿到的模糊需求里已有的约束）。
问题分三类：有效（问到了 hidden 约束）、冗余（只问了已知或已经问过的约束）、不必要（没问到任何约束，
比如问用户根本没要求的颜色；对照组的过度提问主要是这一类）。
论文的多轮设置没有“删掉哪些约束”的记录：hidden 取开场白里没说出来的约束（LLM 判断，有缓存）；
买家按官方做法知道标准答案的规格选项，“忠于需求”的检查只看需求原文会误判，所以这一设置不做买家回答检查。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat_json  # noqa: E402
from shopper_check import judge_answer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = "deepseek-v4-flash"

TARGET_SYSTEM = """给你一条网购需求拆出的约束清单，和购物助手问用户的一个问题。判断这个问题在问用户哪些约束，也就是问题要用户回答或选择的那一项：
- 问"想要哪个颜色""要几个"时，就是在问颜色、数量那条约束，哪怕约束的具体值没有出现在问题里。
- 问题里介绍商品的文字（品类、功能、价格等），如果没有要用户回答，不算问到。
- 请用户确认某个具体规格或商品"可以吗"时，算问到这个规格里体现的约束（如颜色、尺码、数量）。
- 问的方面在清单里没有对应的约束（比如清单里没有尺寸要求，助手却问尺寸），就不算问到任何约束。
输出 JSON：{"targets": ["约束编号", ...]}，一个都没问到就输出空列表。"""

STATED_SYSTEM = """给你一条网购需求拆出的约束清单，和用户对购物助手说的第一句话。判断哪些约束在这句话里已经说出来了：
- 这句话明确表达了这条约束的具体要求，就算说出来了，不要求和原文用词一致。
- 只提到了大类或方向、没说到这条约束的具体要求，不算。
输出 JSON：{"stated": ["约束编号", ...]}，一条都没说就输出空列表。"""


def paper_multiturn(rubric: dict, opening: str) -> dict:
    """论文设置下每题的“信息缺失”记录：开场白没说出来的约束都算 hidden。"""
    listing = "\n".join(f"{c['id']}. {c['description']}" for c in rubric["constraints"])
    out = chat_json(MODEL, STATED_SYSTEM, f"约束清单：\n{listing}\n\n用户的第一句话：{opening}", max_tokens=512)
    stated = set(out.get("stated") or [])
    hidden = [{k: c[k] for k in ("id", "description", "hardness")} for c in rubric["constraints"] if c["id"] not in stated]
    return {"group": "paper", "hidden": hidden, "stated": sorted(stated)}


def load_multiturn() -> dict:
    recs = {}
    for path in (ROOT / "data" / "multiturn").glob("*.jsonl"):
        if path.stem.endswith("_rejected"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["task_id"]] = r
    return recs


def question_targets(rubric: dict, question: str) -> list:
    listing = "\n".join(f"{c['id']}. {c['description']}" for c in rubric["constraints"])
    out = chat_json(MODEL, TARGET_SYSTEM, f"约束清单：\n{listing}\n\n助手的问题：{question}", max_tokens=512)
    valid = {c["id"] for c in rubric["constraints"]}
    return [t for t in out.get("targets") or [] if t in valid]


def analyze(traj, rubric: dict, mt: dict, check: list, acceptable_core: bool, check_answers: bool = True) -> dict:
    hidden = [h["id"] for h in mt["hidden"]]
    kinds = {c["id"]: c for c in rubric["constraints"]}
    verdict = {r["id"]: r.get("verdict") for r in check or []}
    questions, asked, history = [], set(), []
    for q in traj.asks:
        targets = question_targets(rubric, q["question"])
        if set(targets) & (set(hidden) - asked):
            kind = "effective"
        elif targets:
            kind = "redundant"
        else:
            kind = "unnecessary"
        quality = judge_answer(traj.query, traj.agent_query, history, q["question"], q["answer"]) if check_answers else None
        questions.append({"question": q["question"], "answer": q["answer"], "targets": targets, "kind": kind, "answer_check": quality})
        asked.update(targets)
        history.append({"question": q["question"], "answer": q["answer"]})

    hidden_hard = [h for h in hidden if kinds.get(h, {}).get("hardness") == "hard"]
    hidden_asked = [h for h in hidden if h in asked]
    # 被删的 hard 约束在最终购买里没满足（违反或无法确认）
    unmet = [h for h in hidden_hard if traj.purchased and verdict.get(h) != "satisfied"]
    ask_failure = None
    if traj.can_ask and mt["group"] in ("missing", "paper") and traj.purchased and not acceptable_core and unmet:
        ask_failure = "asked_not_used" if all(h in asked for h in unmet) else "should_have_asked"
    used = [h for h in hidden_asked if traj.purchased and verdict.get(h) == "satisfied"]
    return {
        "group": mt["group"],
        "hidden": hidden,
        "hidden_hard": hidden_hard,
        "n_questions": len(traj.asks),
        "questions": questions,
        "hidden_asked": hidden_asked,
        "hidden_asked_used": used,
        "hidden_unmet": unmet,
        "ask_failure": ask_failure,
    }

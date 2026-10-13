"""任务状态层的需求表准不准：用户在对话里说出了哪些约束、需求表收录了没有，表里的条目能不能对上需求原文的约束。

评测侧能看到完整需求拆出的约束清单（data/rubrics/），agent 侧看不到。只看每题最后一次发给模型的状态卡里的需求表，
也就是 agent 实际看到的那一版（原话核对没通过的条目已经被丢掉）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat_json  # noqa: E402
from traj import load_jsonl  # noqa: E402

MODEL = "deepseek-v4-flash"

SYSTEM = """给你一条网购需求拆出的约束清单、用户在对话里说过的全部话，和系统整理出的“用户需求表”。逐条判断：
- 对清单里的每条约束：said 表示用户在对话里有没有说出这条约束的具体要求（只看用户的话，意思对上就算）；in_table 表示需求表里有没有一行表达了同样的要求（措辞可以不同）。
- 对需求表的每一行（R 开头）：match 填它对应的约束编号，对不上任何约束就填空字符串（比如用户随口说的、清单里没有的）；wrong 表示这一行记的内容和用户的话矛盾（记错了）。
输出 JSON：{"constraints": [{"id": "c1", "said": true, "in_table": true}], "rows": [{"row": "R1", "match": "c1", "wrong": false}]}"""


def requirement_table(card: str) -> list:
    """状态卡里“需求表：”下面缩进的那几行。"""
    lines, inside = [], False
    for line in card.splitlines():
        if line.startswith("需求表："):
            inside = True
            continue
        if inside:
            if not line.startswith("  "):
                break
            lines.append(line.strip())
    return lines


def analyze(traj, rubric: dict):
    cards = load_jsonl(traj.dir / "state_cards.jsonl")
    if not cards:
        return None
    rows = [line for line in requirement_table(cards[-1]["card"]) if line.startswith("R")]
    user = [traj.agent_query] + [q["answer"] for q in traj.asks]
    listing = "\n".join(f"{c['id']}. {c['description']}" for c in rubric["constraints"])
    said = "\n".join(f"- {u}" for u in user)
    table = "\n".join(rows) if rows else "（空）"
    out = chat_json(MODEL, SYSTEM, f"约束清单：\n{listing}\n\n用户说过的话：\n{said}\n\n用户需求表：\n{table}", max_tokens=2048)
    valid = {c["id"] for c in rubric["constraints"]}
    cons = [c for c in out.get("constraints") or [] if c.get("id") in valid]
    row_ids = {r.split(" ", 1)[0] for r in rows}
    judged = [r for r in out.get("rows") or [] if r.get("row") in row_ids]
    return {
        "n_rows": len(rows),
        "said": [c["id"] for c in cons if c.get("said")],
        "said_in_table": [c["id"] for c in cons if c.get("said") and c.get("in_table")],
        "rows_matched": sum(1 for r in judged if r.get("match") in valid and not r.get("wrong")),
        "rows_wrong": sum(1 for r in judged if r.get("wrong")),
    }

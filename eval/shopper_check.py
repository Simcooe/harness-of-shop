#!/usr/bin/env python3
"""模拟买家质检。

judge_answer() 判断模拟买家的一次回答：是否针对问题（relevant）、是否忠于完整需求（faithful，
编造需求里没有的要求、或需求里写了却说没要求，都算不忠实）、是否主动透露了没被问到的需求（leaked）。
evaluate.py 用它检查实际对话；直接运行本脚本则在开发集上做上线前的探针测试：
  - 定向问题：针对每题第一条被删掉的规格约束提问，应当答出需求里的内容，而且不透露其他被删的约束；
  - 无关问题：问一项完整需求里没提到的内容，应当回答“没有特别要求”；
  - 选择题：把 gold 商品的规格选项列给买家让它挑（agent 常这样问），选项之间需求没写到的差别不能替自己挑。

用法：python3 eval/shopper_check.py --split dev
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker import load_products  # noqa: E402
from llm import chat_json, chat_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SHOPPER_PROMPT = ROOT / "prompts" / "shopper.md"
SHOPPER_MODEL = "qwen3.7-plus"
JUDGE_MODEL = "deepseek-v4-flash"

JUDGE_SYSTEM = """你在检查一个扮演网购用户的模拟器有没有守规矩。给你用户的完整需求、用户一开始说的话、之前的对话、购物助手这次的问题和模拟用户的回答。逐项判断：
1. relevant：回答是否针对这次的问题。答非所问为 false。
2. faithful：回答的内容是否都有完整需求作依据。说了完整需求里没有的具体要求（编造），或者需求里写了却说没有要求、说错了内容，都是 false。需求里确实没写、回答"没有特别要求/都可以"，是 true。用户用完整需求里的说法说出自己的要求，即使和助手列出的选项都对不上，也是 true。助手让用户从几个选项里挑时，用户挑定了某个选项，而这个选项包含需求里没写到的具体内容（比如需求没提尺寸，用户却挑了带某个尺寸的选项），或者从需求推断错了，才算编造，是 false。
3. leaked：回答是否主动透露了这次问题完全没涉及的其他方面的需求。回答说的是这次问到的那一项（包括助手列出的选项涉及的方面）的具体要求，不算泄露，哪怕用户一开始没说过。
输出 JSON：{"relevant": true 或 false, "faithful": true 或 false, "leaked": true 或 false, "note": "一句话说明"}"""

QUESTION_SYSTEM = """把一条购物需求信息改写成购物助手会问用户的一个问题：问的是这一项用户要什么，但问题里不能出现具体的值。
例如"颜色为白色"改写成"请问您对颜色有什么要求吗？"；"购买数量为两张"改写成"请问您需要买几张？"。
只输出 JSON：{"question": "问题"}"""

UNASKED_DIMENSIONS = [
    ("品牌", ["牌"]),
    ("发货时间", ["发货", "物流", "快递", "邮"]),
    ("产地", ["产", "进口", "国"]),
    ("包装", ["装", "盒"]),
]


def shopper_system(full_query: str, initial_query: str) -> str:
    text = SHOPPER_PROMPT.read_text(encoding="utf-8").strip()
    return text.replace("{full_query}", full_query).replace("{initial_query}", initial_query)


def ask(full_query: str, initial_query: str, history: list, question: str, model: str = SHOPPER_MODEL) -> str:
    messages = [{"role": "system", "content": shopper_system(full_query, initial_query)}]
    for h in history:
        messages += [{"role": "user", "content": h["question"]}, {"role": "assistant", "content": h["answer"]}]
    messages.append({"role": "user", "content": question})
    return chat_text(model, messages)


def judge_answer(full_query: str, initial_query: str, history: list, question: str, answer: str) -> dict:
    past = "\n".join(f"助手：{h['question']}\n用户：{h['answer']}" for h in history) or "（无）"
    user = (f"完整需求：{full_query}\n用户一开始说的话：{initial_query}\n之前的对话：\n{past}\n\n"
            f"助手这次的问题：{question}\n模拟用户的回答：{answer}")
    out = chat_json(JUDGE_MODEL, JUDGE_SYSTEM, user, max_tokens=512)
    return {k: out.get(k) for k in ("relevant", "faithful", "leaked", "note")}


def choice_question(options: dict) -> str:
    parts = []
    for axis, values in options.items():
        names = [str(v["value"]) for v in values][:8]
        if len(names) >= 2:
            parts.append(f"{axis}有：{' / '.join(names)}")
    return f"这款商品的规格{'；'.join(parts)}。请问您要哪一个？" if parts else ""


def probe(rec: dict, products: dict, gold_asin: str) -> list:
    rows = []
    hidden = rec["hidden"][0]
    question = chat_json(JUDGE_MODEL, QUESTION_SYSTEM, f"信息：{hidden['description']}", max_tokens=256).get("question", "")
    dim = next((d for d, keys in UNASKED_DIMENSIONS if not any(k in rec["full_query"] for k in keys)), None)
    probes = [("定向", question)] + ([("无关", f"请问{dim}有什么要求吗？")] if dim else [])
    choice = choice_question((products.get(gold_asin) or {}).get("options") or {})
    if choice:
        probes.append(("选择题", choice))
    for kind, q in probes:
        answer = ask(rec["full_query"], rec["vague_query"], [], q)
        verdict = judge_answer(rec["full_query"], rec["vague_query"], [], q, answer)
        rows.append({"task_id": rec["task_id"], "kind": kind, "question": q, "answer": answer, "hidden": hidden["description"], **verdict})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    recs = [json.loads(l) for l in (ROOT / "data" / "multiturn" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r["group"] == "missing"]
    products = load_products()
    gold = {json.loads(l)["task_id"]: str(json.loads(l)["asin"]) for l in (ROOT / "data" / "private" / "gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for rows in pool.map(lambda r: probe(r, products, gold[r["task_id"]]), recs) for row in rows]
    out = ROOT / "data" / "cache" / f"shopper_probe_{args.split}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    for kind in ("定向", "无关", "选择题"):
        sub = [r for r in rows if r["kind"] == kind]
        c = Counter()
        for r in sub:
            c["relevant"] += r["relevant"] is True
            c["faithful"] += r["faithful"] is True
            c["leaked"] += r["leaked"] is True
        print(f"{kind}问题 {len(sub)} 个：针对问题 {c['relevant']}，忠于需求 {c['faithful']}，泄露其他需求 {c['leaked']}")
    bad = [r for r in rows if r["relevant"] is not True or r["faithful"] is not True or r["leaked"] is True]
    print(f"有问题的回答 {len(bad)} 个：")
    for r in bad[:25]:
        print(f"  [{r['task_id']} {r['kind']}] 问：{r['question']} | 答：{r['answer']} | {r['note']}")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()

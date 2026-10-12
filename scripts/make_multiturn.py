#!/usr/bin/env python3
"""构造多轮实验用的“模糊需求”：从冻结的约束清单里取出规格类约束（颜色、尺码、容量、数量、套餐等），
让 LLM 把它们从需求里删掉，再用规则 + LLM 校验，结果冻结到 data/multiturn/<split>.jsonl。

每题三份信息：vague_query（agent 看）、full_query（模拟买家看）、hidden（被删的约束，评测“该问什么”的依据）。
不带规格约束的题原样保留，作为“不需要问”的对照组。

用法：python3 scripts/make_multiturn.py --split dev
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from llm import chat_json  # noqa: E402

GOLD_PATH = ROOT / "data" / "private" / "gold.jsonl"
RUBRIC_DIR = ROOT / "data" / "rubrics"
OUT_DIR = ROOT / "data" / "multiturn"
MODEL = "deepseek-v4-flash"
MAX_ATTEMPTS = 3

REWRITE_SYSTEM = """你负责改写网购需求，造出"信息不完整"的版本，用来测试购物助手会不会主动向用户提问。

给你一条需求和要删掉的信息，输出删掉之后的需求：
1. 彻底删掉列出的信息：具体的值和指向这一项的说法都不要留。比如删"颜色为米白色"时，"米白色"和"颜色"都不留；删"1瓶60毫升"时，"60毫升"和"1瓶"都不留。
2. 不要换成别的说法，也不要留暗示，比如不要写"颜色待定""某个尺寸""规格随意"。
3. 没要求删的内容尽量保留原话，只做让句子通顺所必需的最小改动。
4. 不要加入任何新信息。
只输出 JSON：{"vague_query": "改写后的需求"}"""

CHECK_SYSTEM = """你是数据质检员。给你一条网购需求和一组信息点，逐条判断：只看这条需求，能不能知道这个信息点的具体内容。
明说了，或者能直接推断出具体内容，都算知道；只能看出"有这么一项"但不知道具体是什么，不算知道。
输出 JSON：{"results": [{"id": "信息点编号", "known": true 或 false, "evidence": "需求中的相关片段，没有就留空"}]}"""


def squash(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", str(text or "")))


def has_number(text: str, num: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(num)}(?![\d.])", text) is not None


def rewrite(query: str, hidden: list, feedback: str) -> str:
    lines = [f"需求：{query}", "要删掉的信息："]
    lines += [f"- {c['description']}（原文：“{c['query_quote']}”）" for c in hidden]
    if feedback:
        lines += ["", f"上一次改写的问题：{feedback}", "请重新改写。"]
    out = chat_json(MODEL, REWRITE_SYSTEM, "\n".join(lines), max_tokens=1024)
    return str(out.get("vague_query") or "").strip()


def check(vague: str, query: str, hidden: list, kept: list) -> list:
    """返回问题列表，空列表表示通过。"""
    problems = []
    if not vague or squash(vague) == squash(query):
        return ["改写结果为空或和原需求一样"]
    kept_numbers = set().union(*(numbers(c["query_quote"]) for c in kept)) if kept else set()
    for c in hidden:
        if squash(c["query_quote"]) in squash(vague):
            problems.append(f"原文片段“{c['query_quote']}”还在")
        for num in numbers(c["query_quote"]) - kept_numbers:
            if has_number(vague, num):
                problems.append(f"“{c['description']}”里的数字 {num} 还在")
    items = hidden + kept
    listing = "\n".join(f"{c['id']}. {c['description']}" for c in items)
    out = chat_json(MODEL, CHECK_SYSTEM, f"需求：{vague}\n\n信息点：\n{listing}", max_tokens=2048)
    known = {str(r.get("id")): r for r in out.get("results") or []}
    for c in hidden:
        r = known.get(c["id"])
        if r is None or r.get("known") is not False:
            problems.append(f"还能看出“{c['description']}”（{(r or {}).get('evidence', '质检没有给出结论')}）")
    for c in kept:
        r = known.get(c["id"])
        if c["hardness"] == "hard" and r is not None and r.get("known") is False:
            problems.append(f"误删了“{c['description']}”")
    return problems


def build(task: dict) -> dict:
    rubric = json.loads((RUBRIC_DIR / f"{task['task_id']}.json").read_text(encoding="utf-8"))
    query = rubric["query"]
    hidden = [c for c in rubric["constraints"] if c["type"] == "spec"]
    kept = [c for c in rubric["constraints"] if c["type"] != "spec"]
    record = {"task_id": task["task_id"], "split": task["split"], "full_query": query}
    if not hidden:
        return {**record, "group": "control", "vague_query": query, "hidden": [], "valid": True, "attempts": 0, "problems": []}
    feedback, vague, problems = "", "", []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        vague = rewrite(query, hidden, feedback)
        problems = check(vague, query, hidden, kept)
        if not problems:
            break
        feedback = "；".join(problems)
    hidden_out = [{k: c[k] for k in ("id", "description", "query_quote", "hardness")} for c in hidden]
    return {**record, "group": "missing", "vague_query": vague, "hidden": hidden_out,
            "valid": not problems, "attempts": attempt, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = [t for t in tasks if t["split"] == args.split]
    order = json.loads((ROOT / "data" / f"{args.split}_ids.json").read_text())
    rank = {tid: i for i, tid in enumerate(order)}
    tasks.sort(key=lambda t: rank[t["task_id"]])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(build, tasks))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.split}.jsonl"
    valid = [r for r in records if r["valid"]]
    with out.open("w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rejected = [r for r in records if not r["valid"]]
    (OUT_DIR / f"{args.split}_rejected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")

    missing = [r for r in valid if r["group"] == "missing"]
    print(f"{args.split}：{len(records)} 题；信息缺失组 {len(missing)} 题，对照组 {len(valid) - len(missing)} 题；"
          f"校验未通过、剔除 {len(rejected)} 题 {[r['task_id'] for r in rejected]}")
    print(f"  改写次数分布：{ {n: sum(1 for r in missing if r['attempts'] == n) for n in range(1, MAX_ATTEMPTS + 1)} }")
    print(f"  写入 {out}")


if __name__ == "__main__":
    main()

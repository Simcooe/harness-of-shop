#!/usr/bin/env python3
"""画像设置的分析（只用来分析错误，不进成绩）。

1. 信息在哪里：画像设置里用户只知道简版需求和规格选择，其余细节记在用户画像里（画像里也混着很多和这次购买无关的长期偏好）。
   每条 hard 约束由评审模型（V4 Pro，有缓存）判断信息在哪里：用户知道 / 画像的哪一部分（近期行为、用户标签、长期偏好、其他）/ 哪里都没有，
   再看买到的商品在每组约束上满足得怎么样。
2. --shopper：模拟用户质检。用户只知道简版需求和规格选择，检查它的每次回答有没有把这之外的东西当成购买条件、或者编出具体要求
   （V4 Flash，有缓存）。

用法：python3 eval/persona.py runs/<run> [--shopper]   （先跑 eval/evaluate.py）
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker import load_rubric  # noqa: E402
from judge import JUDGE_MODEL  # noqa: E402
from llm import chat_json  # noqa: E402
from traj import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "private" / "gold.jsonl"
SECTION_NAMES = {"recent": "近期行为", "tags": "用户标签", "prefs": "长期偏好", "other": "其他"}
SHOPPER_JUDGE_MODEL = "deepseek-v4-flash"

SHOPPER_SYSTEM = """你在检查一个扮演网购用户的模拟器有没有守规矩。这个用户只知道自己的购买任务和规格选择，其余细节它并不知道。给你它知道的需求、购物助手这次说的话和用户的回答。判断：
- invented_condition：用户是否把购买任务和规格选择之外的东西当成了必须满足或必须确认的条件（比如要求先确认品牌、某个功能参数，否则不买）。只是顺口问一句、不作为条件，不算。
- invented_value：用户是否说出了购买任务和规格选择之外的具体要求（比如自己编了一个品牌、尺寸、功能）。
输出 JSON：{"invented_condition": true 或 false, "invented_value": true 或 false, "note": "一句话说明"}"""

SYSTEM = """你在分析一道网购测试题。用户的完整需求被拆成了若干条约束。对话里的用户自己只知道两样：简版需求和规格选择；其余细节记在平台的用户个人资料里，个人资料里也混着很多和这次购买无关的长期偏好。
对每条约束，按下面的顺序判断它的信息在哪里能找到（说法不同、意思一样也算找到）：
1. 先看简版需求和规格选择：只要其中有，就是 user，不管个人资料里有没有；
2. 两处都没有，再看个人资料：有就是 profile，section 写在资料的哪一部分：recent（最近14天的搜索、加购、收藏、关注的店铺）、tags（用户标签）、prefs（兴趣偏好：品牌、类目、商品属性偏好）、other（其他字段）；
3. 哪里都没有就是 none。
evidence 逐字摘抄找到它的那一小段：user 摘简版需求或规格选择，profile 摘个人资料，none 留空。
输出 JSON：{"sources": [{"id": "c1", "source": "user|profile|none", "section": "recent|tags|prefs|other 或留空", "evidence": ""}]}"""


def sources(rubric: dict, shopper_query: str, goal_options, profile: dict) -> dict:
    cons = "\n".join(f"{c['id']} [{c['type']}] {c['description']}（需求原文：{c['query_quote']}）" for c in rubric["constraints"])
    user = (f"简版需求：{shopper_query}\n规格选择：{json.dumps(list(goal_options or []), ensure_ascii=False)}\n"
            f"用户个人资料：{json.dumps(profile, ensure_ascii=False)}\n\n约束：\n{cons}")
    raw = chat_json(JUDGE_MODEL, SYSTEM, user, max_tokens=3000)
    out = {}
    for s in raw.get("sources") or []:
        if s.get("source") == "user":
            out[s.get("id")] = "用户知道"
        elif s.get("source") == "profile":
            out[s.get("id")] = f"画像·{SECTION_NAMES.get(s.get('section'), '其他')}"
        else:
            out[s.get("id")] = "哪里都没有"
    return out


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def analyze(run: Path, rec: dict, gold: dict):
    d = run / f"task_{rec['task_id']}"
    if not (d / "persona.json").exists():
        return None
    profile = json.loads((d / "persona.json").read_text(encoding="utf-8"))
    reset = json.loads((d / "reset_public.json").read_text(encoding="utf-8"))
    rubric = load_rubric(rec["task_id"])
    return rubric, sources(rubric, reset["shopper_query"], gold[rec["task_id"]]["options"], profile)


def check_shopper(task_dir: Path, gold: dict) -> list:
    reset = json.loads((task_dir / "reset_public.json").read_text(encoding="utf-8"))
    tid = int(task_dir.name.split("_")[-1])
    known = f"购买任务：{reset['shopper_query']}\n规格选择：{json.dumps(gold[tid]['options'], ensure_ascii=False)}"
    out = []
    for r in load_jsonl(task_dir / "ask_log.jsonl"):
        raw = chat_json(SHOPPER_JUDGE_MODEL, SHOPPER_SYSTEM, f"{known}\n\n助手：{r['question'][:1500]}\n\n用户：{r['answer']}", max_tokens=400)
        out.append({"task_id": tid, "turn": r.get("turn"), "condition": bool(raw.get("invented_condition")),
                    "value": bool(raw.get("invented_value")), "note": raw.get("note", "")})
    return out


def shopper_report(run: Path, gold: dict) -> None:
    dirs = sorted(d for d in run.glob("task_*") if (d / "persona.json").exists())
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = [a for rows in pool.map(lambda d: check_shopper(d, gold), dirs) for a in rows]
    cond = [a for a in answers if a["condition"]]
    value = [a for a in answers if a["value"]]
    print(f"{len(dirs)} 题，用户回答 {len(answers)} 次：把任务外的东西当成购买条件 {len(cond)} 次（{len({a['task_id'] for a in cond})} 题），"
          f"编出任务外的具体要求 {len(value)} 次（{len({a['task_id'] for a in value})} 题）")
    for a in cond:
        print(f"  task {a['task_id']} 第 {a['turn']} 回合：{a['note']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--shopper", action="store_true", help="模拟用户质检")
    args = parser.parse_args()
    run = args.run_dir
    gold = {row["task_id"]: row for row in load_jsonl(GOLD_PATH)}
    if args.shopper:
        shopper_report(run, gold)
        return
    recs = load_jsonl(run / "eval" / "tasks.jsonl")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda r: analyze(run, r, gold), recs))
    where, verdicts, failures = Counter(), {}, []
    for r, res in zip(recs, results):
        if res is None:
            continue
        rubric, src = res
        hard = {c["id"]: c for c in rubric["constraints"] if c["hardness"] == "hard"}
        where.update(src.get(cid, "?") for cid in hard)
        if not r.get("purchased"):
            failures.append((r["task_id"], "没买", []))
            continue
        for c in r.get("check") or []:
            if c["id"] in hard:
                verdicts.setdefault(src.get(c["id"], "?"), Counter())[c["verdict"]] += 1
        if not r["strict_success"]:
            bad = [f"{src.get(c['id'], '?')}：{hard[c['id']]['description']}" for c in r.get("check") or []
                   if c["id"] in hard and c["verdict"] == "violated"]
            failures.append((r["task_id"], "买错" + ("（买到的可接受）" if r.get("acceptable_core") else ""), bad))

    total = sum(where.values())
    print(f"hard 约束 {total} 条，信息在哪里：" + "，".join(f"{k} {pct(v, total)}" for k, v in where.most_common()))
    print("买下的商品满足了多少（按信息在哪里分组）：")
    for name, cnt in sorted(verdicts.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(cnt.values())
        print(f"  {name:<8} {n} 条：满足 {pct(cnt['satisfied'], n)}，违反 {pct(cnt['violated'], n)}，看不出 {pct(cnt['unknown'], n)}")
    print(f"没有严格成功 {len(failures)} 题：")
    for tid, kind, bad in failures:
        print(f"  task {tid} {kind}" + (f"；违反：{'；'.join(bad)}" if bad else ""))


if __name__ == "__main__":
    main()

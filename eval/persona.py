#!/usr/bin/env python3
"""画像设置的分析（只用来分析错误，不进成绩）。

1. 信息在哪里：画像设置里用户只知道简版需求和规格选择，其余细节记在用户画像里（画像里也混着很多和这次购买无关的长期偏好）。
   每条 hard 约束由评审模型（eval/judge.py 的 JUDGE_MODEL，有缓存）判断信息在哪里：用户知道 / 画像的哪一部分（近期行为、用户标签、长期偏好、其他）/ 哪里都没有，
   再看买到的商品在每组约束上满足得怎么样。
2. 针对画像设置里的两类错各给一个数：用户点头了但买错的题；买手拿画像里和这次无关的牌子去搜的题。
3. 同款重复上架：买到的和正确商品是同一款（同型号同规格，只是不同店铺上架），严格成功判错，但不是导购能分辨的。
   由评审模型（同上）判断，单列出来，不改严格成功。
4. --shopper：模拟用户质检。用户只知道简版需求和规格选择，检查它的每次回答有没有把这之外的东西当成购买条件、或者编出具体要求
   （V4 Flash，有缓存）。

用法：python3 eval/persona.py runs/<run> [--shopper]   （先跑 eval/evaluate.py）
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker import load_products, load_rubric  # noqa: E402
from judge import JUDGE_MODEL  # noqa: E402
from llm import chat_json  # noqa: E402
from traj import Trajectory, load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "private" / "gold.jsonl"
SECTION_NAMES = {"recent": "近期行为", "tags": "用户标签", "prefs": "长期偏好", "other": "其他"}
SHOPPER_JUDGE_MODEL = "deepseek-v4-flash"

DUP_SYSTEM = """给你两件网购商品的信息（标题、店铺、类目、属性）和各自选的规格。判断它们是不是同一款商品被不同店铺重复上架：
- 两件的品牌一样，型号也一样（标题里的型号、系列编号对得上），或者两个标题几乎逐字一样；
- 并且选的规格一样（写法可以略有不同）。
只是品类、功能、风格相近，看不出是同一个品牌型号的，不算同一款；拿不准就判 false。
输出 JSON：{"brand_a": "", "model_a": "", "brand_b": "", "model_b": "", "same": true 或 false, "reason": "一句话"}"""

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


def _key(text) -> str:
    return re.sub(r"[\s\-_/·]", "", str(text)).lower()


def brand_searches(task_dir: Path, profile: dict, full_query: str, gold_text: str) -> dict:
    """买手的搜索词里用了画像里哪些牌子（不算完整需求里本来就有的）；其中哪些不是正确商品的牌子。"""
    brands = [b.get("品牌名称", "") for b in ((profile.get("兴趣偏好") or {}).get("品牌偏好") or []) if isinstance(b, dict)]
    queries = [s["args"].get("query", "") for s in load_jsonl(task_dir / "env_steps.jsonl")
               if s.get("tool") == "search_products" and s.get("who") != "下单"]
    used = {b for b in brands for q in queries if _key(b) and _key(b) in _key(q) and _key(b) not in _key(full_query)}
    return {"used": sorted(used), "unrelated": sorted(b for b in used if _key(b) not in gold_text)}


UNIT = re.compile(r"^\d+(?:\.\d+)?(?:ml|g|kg|cm|mm|m|l|w|v|a|mah|hz|p)$")


def _codes(text) -> set:
    """标题和规格里的型号编号（字母和数字混在一起的串，去掉 2ml、100g 这类单位）。"""
    return {c for c in re.findall(r"[a-z0-9]+", str(text).lower()) if re.search(r"[a-z]", c) and re.search(r"\d", c) and not UNIT.match(c)}


def _jaccard(a: str, b: str) -> float:
    sa, sb = {a[i:i + 2] for i in range(len(a) - 1)}, {b[i:i + 2] for i in range(len(b) - 1)}
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def same_listing(products: dict, bought: dict, gold: dict) -> dict:
    """评审模型说是同一款，而且标题几乎一样或者有同一个型号编号，才算同款重复上架。"""
    def card(p: dict, options) -> str:
        return (f"标题：{p.get('title', '')}\n店铺：{p.get('shop', '')}\n类目：{p.get('category', '')}\n"
                f"属性：{'；'.join(p.get('attributes') or [])}\n选的规格：{json.dumps(options, ensure_ascii=False)}")
    a, b = products.get(str(bought.get("asin")), {}), products.get(str(gold["asin"]), {})
    raw = chat_json(JUDGE_MODEL, DUP_SYSTEM, f"商品一：\n{card(a, bought.get('options'))}\n\n商品二：\n{card(b, gold['options'])}", max_tokens=300)
    text_a = f"{a.get('title', '')} {json.dumps(bought.get('options'), ensure_ascii=False)}"
    text_b = f"{b.get('title', '')} {json.dumps(gold['options'], ensure_ascii=False)}"
    close = _jaccard(_key(a.get("title", "")), _key(b.get("title", ""))) >= 0.6 or bool(_codes(text_a) & _codes(text_b))
    return {"same": bool(raw.get("same")) and close, "judge_same": bool(raw.get("same")), "reason": raw.get("reason", "")}


def targeted(run: Path, rec: dict, gold: dict, products: dict):
    """两类画像错误和同款重复上架，逐题。"""
    d = run / f"task_{rec['task_id']}"
    if not (d / "persona.json").exists():
        return None
    profile = json.loads((d / "persona.json").read_text(encoding="utf-8"))
    reset = json.loads((d / "reset_public.json").read_text(encoding="utf-8"))
    g = gold[rec["task_id"]]
    gold_text = _key(g["title"]) + _key(products.get(str(g["asin"]), {}).get("shop", ""))
    out = {"agreed_wrong": bool(rec.get("purchased") and not rec["strict_success"] and (rec.get("consent") or {}).get("attitude") == "agree"),
           "brands": brand_searches(d, profile, reset["query"], gold_text), "same_listing": None}
    purchase = Trajectory(d).purchase
    if purchase.get("asin") and not rec["strict_success"] and str(purchase["asin"]) != str(g["asin"]):
        out["same_listing"] = same_listing(products, purchase, g)
    return out


def targeted_report(recs: list, rows: list) -> None:
    pairs = [(r, t) for r, t in zip(recs, rows) if t]
    n = len(pairs)
    agreed_wrong = [r for r, t in pairs if t["agreed_wrong"]]
    dup = [r for r, t in pairs if (t["same_listing"] or {}).get("same")]
    print(f"用户点头了但买错：{pct(len(agreed_wrong), n)}，其中同款重复上架 {sum(1 for r in agreed_wrong if r in dup)} 题")
    unrelated = [(r, t) for r, t in pairs if t["brands"]["unrelated"]]
    right = [(r, t) for r, t in pairs if t["brands"]["used"] and not t["brands"]["unrelated"]]
    print(f"买手拿画像里和这次无关的牌子去搜：{pct(len(unrelated), n)}，其中没买对 {sum(1 for r, _ in unrelated if not r['strict_success'])} 题；"
          f"拿画像里正好对的牌子去搜 {len(right)} 题")
    for r, t in unrelated:
        print(f"  task {r['task_id']}：{'、'.join(t['brands']['unrelated'])}（{'对' if r['strict_success'] else '错'}）")
    strict = sum(r["strict_success"] for r, _ in pairs)
    print(f"同款重复上架（买到的和正确商品是同一款，只是不同店铺）：{len(dup)} 题 {[r['task_id'] for r in dup]}；"
          f"这些也算对的话，严格成功 {pct(strict + len(dup), n)}（参考，正式成绩不变）")


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

    products = load_products()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda r: targeted(run, r, gold, products), recs))
    targeted_report(recs, rows)


if __name__ == "__main__":
    main()

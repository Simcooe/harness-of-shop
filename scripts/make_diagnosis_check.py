#!/usr/bin/env python3
"""失败原因诊断的人工抽查清单：从几次运行的诊断结果里按原因分层抽题，每题列出需求、买了什么、规则判定的出错步骤、
模型判断的原因和理由，以及按回合精简的轨迹，请人判断原因对不对。

用法：python3 scripts/make_diagnosis_check.py runs/dev_paper_v2 runs/dev_paper_v3_w0 ... --out docs/diagnosis_check.md
（这些运行要先跑过 eval/evaluate.py --diagnose；只用开发集的运行，测试集不拿来调评测）
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from checker import load_products  # noqa: E402
from diagnose import CAUSES, TAGS  # noqa: E402
from evaluate import STAGE_NAMES  # noqa: E402
from traj import Trajectory, load_jsonl  # noqa: E402

PER_CAUSE = {"没问清需求": 3, "看到过正确商品没回头": 4, "搜索词不对": 2, "规格没点": 1, "规格选错": 1, "标准答案有问题": 1, "替代品也满足需求": 1}
MOVES = {"next_page": "翻到下一页", "prev_page": "返回上一页", "back_to_search": "回到搜索页"}
SUBPAGES = {"view_description": "描述", "view_features": "特点", "view_reviews": "评价"}
# 每类原因成立要满足的事实，拆成能对着轨迹回答的是/否问题
QUESTIONS = {
    "没问清需求": ["理由里列出的条件，用户在对话里确实没说过（或者说的时候已经快到必须下单了）？",
              "正确商品满足这些条件，买的那件不满足或确认不了？"],
    "看到过正确商品没回头": ["用户在理由里说的那一回合，确实说过这条要求？",
                   "之后正确商品确实出现过，而且它满足这条、买的那件不满足或确认不了？"],
    "搜索词不对": ["用户在理由里说的那一回合，确实说过这条要求，而正确商品满足它？",
              "之后正确商品确实一直没出现在搜索结果里，而按用户说过的话本来想得到能搜出它的词？"],
    "规格没点": ["买的就是正确商品，但下单时有规格项没点（“agent 买的”那栏比正确规格少了一项）？"],
    "规格选错": ["买的就是正确商品，规格点了但和正确规格不一样？", "按用户说过的话或预算，能确定该选正确规格？"],
    "标准答案有问题": ["用户确实说过这条要求，而正确商品不满足、买的那件满足？"],
    "替代品也满足需求": ["按需求一条条看，买的那件确实不比正确商品差？"],
}


def short(text, n=80) -> str:
    text = str(text or "").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def describe_step(s: dict, gold_asin: str) -> str:
    tool, args = s.get("tool"), s.get("args") or {}
    result = s.get("result") or {}
    st = result.get("observation_state") or {}
    if tool == "buy_now" and result.get("done"):
        return f"S{s.get('step')} 下单"
    if st.get("page_type") == "search_results":
        asins = [str(p.get("asin")) for p in st.get("products") or []]
        where = f"，正确商品在第 {asins.index(gold_asin) + 1} 个" if gold_asin in asins else ""
        head = f"搜「{args.get('query')}」" if tool == "search_products" else MOVES.get(tool, tool)
        return f"S{s.get('step')} {head} → 这页 {len(asins)} 个商品{where}"
    if tool == "open_product":
        p = st.get("product") or {}
        mark = "【正确商品】" if str(p.get("asin")) == gold_asin else ""
        return f"S{s.get('step')} 打开「{short(p.get('title'), 26)}」{mark}"
    if tool == "select_option":
        return f"S{s.get('step')} 选规格「{args.get('value')}」"
    if tool in SUBPAGES:
        return f"S{s.get('step')} 看{SUBPAGES[tool]}子页"
    if result.get("done"):
        return f"S{s.get('step')} 步数用完"
    return f"S{s.get('step')} {MOVES.get(tool, tool)}"


def harness_note(e: dict, seen: set):
    kind = str(e.get("type", ""))
    if kind in ("closing_hard", "state_close_hard"):
        return None if "hard" in seen else (seen.add("hard") or "〔系统：现在必须下单〕")
    if kind in ("closing_soft", "state_close_soft"):
        return None if "soft" in seen else (seen.add("soft") or f"〔系统：还剩 {e.get('left')} 个回合，请尽快下单〕")
    if kind in ("confirm_block", "state_verify_block"):
        return f"〔系统拦下了这次下单：{short('；'.join(e.get('issues') or []), 120)}〕"
    if kind in ("guard_block", "audit_block"):
        return f"〔系统拦下了 {e.get('tool')}：{short(e.get('reason'), 40)}〕"
    if kind in ("ask_hint", "state_ask"):
        return f"〔系统提示：可以把规格选项列给用户挑（{'、'.join(e.get('axes') or [])}）〕"
    return None


def compact(traj: Trajectory, gold_asin: str) -> list:
    """每个回合一行：agent 的商店动作（正确商品标出来），下面跟着这一回合的对话和系统干预。"""
    turns = defaultdict(lambda: {"steps": [], "talk": []})
    for s in traj.steps:
        turns[s.get("turn") or 0]["steps"].append(describe_step(s, gold_asin))
    for q in traj.asks:
        turns[q.get("turn") or 0]["talk"] += [f"agent：{short(q.get('question'), 90)}", f"用户：{short(q.get('answer'), 120)}"]
    seen = set()
    for e in traj.harness:
        note = harness_note(e, seen)
        if note:
            turns[e.get("turn") or 0]["talk"].append(note)
    lines = []
    for turn in sorted(turns):
        t = turns[turn]
        lines.append(f"- 回合 {turn}：" + ("；".join(t["steps"]) or "（没有商店动作）"))
        lines += [f"  - {x}" for x in t["talk"]]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "diagnosis_check.md")
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    products = load_products()
    gold = {row["task_id"]: row for row in load_jsonl(ROOT / "data" / "private" / "gold.jsonl")}
    pool = defaultdict(list)
    for run in args.runs:
        for r in load_jsonl(run / "eval" / "tasks.jsonl"):
            d = r.get("diagnosis")
            # 只有一个选项的规格没点，日志就能确定，不用抽查
            if d and not (d["cause"] == "规格没点" and all(f.get("single") for f in (d.get("facts") or {}).get("spec") or [])):
                pool[d["cause"]].append((run, r))
    rng = random.Random(args.seed)
    picked = []
    for cause, k in PER_CAUSE.items():
        cands = pool.get(cause, [])
        picked += rng.sample(cands, min(k, len(cands)))

    L = ["# 失败原因诊断抽查（开发集）", "",
         "**这是什么**：每道题都是 agent 没有严格做对的题（没买到标准答案那件，或者规格没全对）。首错定位只说得出第一次出错在哪一步；"
         "这里按几条能核对的事实由规则判定原因：每条需求买的那件和正确商品各自满不满足、用户第几回合说过、正确商品什么时候出现过、规格按用户的话能不能推出来。",
         "", "**怎么做**：每条下面有一两个是/否问题，对着需求、对话和轨迹回答。都是“是”，就判断对了；有“否”的话，在最后一行写你觉得应该是哪一类。",
         "", "**原因的几类**：", *[f"- {k}：{v}" for k, v in CAUSES.items()], "",
         "**附加标签**（耗掉回合的方式等，列在原因后面）：", *[f"- {k}：{v}" for k, v in TAGS.items()], "",
         "系统拦下了正确的购买、正确商品难搜到、没买成、只有一个选项的规格没点，日志就能确定，不在抽查范围。", "",
         f"共 {len(picked)} 条，从 {'、'.join(r.name for r in args.runs)} 的诊断结果里按原因分层随机抽取。轨迹里 S 后面的数字是环境步骤编号。", ""]
    for i, (run, r) in enumerate(picked, 1):
        t = Trajectory(run / f"task_{r['task_id']}")
        g = gold[r["task_id"]]
        gold_asin = str(g["asin"])
        p = t.purchase
        d = r["diagnosis"]
        fe = r.get("first_error") or {}
        bought = f"「{short((products.get(str(p.get('asin'))) or {}).get('title'), 40)}」" if p.get("asin") else "没买"
        tags = f"（附加标签：{'、'.join(d['tags'])}）" if d.get("tags") else ""
        L += [f"## 第 {i} 条：判定「{d['cause']}」{tags}（{run.name}，第 {r['task_id']} 题）", "",
              f"**用户的完整需求**（agent 看不到，只能靠问）：{t.query}", "",
              f"**正确商品**：「{short((products.get(gold_asin) or {}).get('title'), 40)}」，规格 {'、'.join(g.get('options') or []) or '无'}", "",
              f"**agent 买的**：{bought}，规格 {json.dumps(p.get('options') or {}, ensure_ascii=False)}"
              + ("（就是正确商品）" if str(p.get("asin")) == gold_asin else "")
              + ("；评测判定可以接受" if r.get("acceptable_core") else "；评测判定不可接受"), "",
              f"**规则说错在哪一步**：{STAGE_NAMES.get(fe.get('type'), fe.get('type'))}" + (f"（S{fe['step']}）" if fe.get("step") else ""), "",
              *([f"**对照**：知道完整需求的 agent 在 {d['reference']['runs']} 次运行里 {d['reference']['retrieved']} 次搜到过正确商品，"
                 f"用的词：{'、'.join(f'「{q}」' for q in d['reference']['queries'])}", ""] if (d.get("reference") or {}).get("retrieved") else []),
              f"**为什么错**：{d['explanation']}", "",
              "**请回答**：", *[f"{j}. {q}（是/否）" for j, q in enumerate(QUESTIONS.get(d["cause"], []), 1)], "",
              "<details><summary>按回合的精简轨迹</summary>", "",
              f"- 开场：{t.agent_query}", *compact(t, gold_asin), "", "</details>", "",
              "**你的结论**：对 / 不对（应该是：　　　）", ""]
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"写入 {args.out}，共 {len(picked)} 题；候选 {dict((k, len(v)) for k, v in pool.items())}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""生成人工抽查清单：按类别抽 20 条轨迹。每条用大白话写评测器的结论，附上相关的几步轨迹，
再给一两个是/否问题，读完就能判断评测器对不对，不用另外跑命令。

用法：python3 scripts/make_spotcheck.py runs/dev100_off runs/dev100_on [--out docs/spotcheck_dev.md]
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from process import final_words, is_invalid  # noqa: E402
from traj import Trajectory  # noqa: E402

VERDICT = {"satisfied": "✔ 满足", "violated": "✘ 违反", "unknown": "？无法确认"}
VERIFY = {
    "verified": "购买前看到过依据",
    "unverified": "购买前没看到依据",
    "knowingly_violated": "看到过矛盾信息仍然买了",
    "blind_violation": "没看就买，且不满足",
    "no_evidence": "商品信息里没写",
    "not_displayed": "页面不显示价格",
}
PAGE = {"search_home": "搜索首页", "search_results": "搜索结果页", "product_detail": "商品详情页"}
GOLD_PATH = ROOT / "data" / "private" / "gold.jsonl"


def page_name(state: dict) -> str:
    page = state.get("page_type")
    if page == "information_subpage":
        return f"{state.get('subpage', '')} 详情子页"
    return PAGE.get(page, str(page))


def short(text, n=40) -> str:
    text = str(text or "").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def describe_action(step: dict) -> str:
    tool, args = step.get("tool"), step.get("args") or {}
    return {
        "search_products": f"搜索「{args.get('query', '')}」",
        "open_product": f"打开商品 {args.get('asin', '')}",
        "select_option": f"选规格「{args.get('value', '')}」",
        "view_description": "打开 Description 详情子页",
        "view_features": "打开 Features 详情子页",
        "view_reviews": "打开 Reviews 详情子页",
        "next_page": "翻到下一页",
        "prev_page": "返回上一页",
        "back_to_search": "回到搜索首页",
        "buy_now": "点击购买",
    }.get(tool, str(tool))


def step_lines(traj: Trajectory, lo: int, hi: int, gold_asin: str) -> list:
    lines = []
    for i in range(max(0, lo), min(len(traj.steps), hi)):
        s = traj.steps[i]
        before = traj.state_before(i)
        result = s.get("result") or {}
        after = result.get("observation_state") or {}
        if result.get("done"):
            outcome = "✅ 购买成功，任务结束" if s.get("tool") == "buy_now" else "任务结束（步数用完）"
        elif is_invalid(traj, i):
            outcome = f"⚠️ 动作无效，页面没变（仍在{page_name(after)}）"
        else:
            outcome = f"进入{page_name(after)}"
            p = after.get("product") or {}
            if after.get("page_type") in ("product_detail", "information_subpage") and p:
                outcome += f"：{short(p.get('title'), 30)}"
            if str(p.get("asin")) == gold_asin:
                outcome += "（标准答案商品）"
            if after.get("selected_options") and s.get("tool") == "select_option":
                outcome += f"，已选 {after.get('selected_options')}"
        lines.append(f"- 第 {s.get('step')} 步：在{page_name(before)}上{describe_action(s)} → {outcome}")
    return lines


def constraint_table(rubric: dict, rec: dict) -> list:
    check = {c["id"]: c for c in rec.get("check") or []}
    ver = rec.get("verification") or {}
    rows = ["| 用户的要求 | 评测器判断 | 依据（商品信息原文） | 购买前有没有看到 |", "|---|---|---|---|"]
    for c in rubric["constraints"]:
        ck = check.get(c["id"], {})
        quote = short(ck.get("evidence_quote", ""), 26).replace("|", "/")
        soft = "（偏好）" if c["hardness"] == "soft" else ""
        rows.append(f"| {c['description']}{soft} | {VERDICT.get(ck.get('verdict'), '-')} | {quote or '—'} | {VERIFY.get(ver.get(c['id']), '—')} |")
    return rows


def build_item(name, run: Path, rec: dict, rubric: dict, gold: dict) -> list:
    traj = Trajectory(run / f"task_{rec['task_id']}")
    gold_asin = str(gold["asin"])
    n = len(traj.steps)
    words = short(final_words(traj), 300)
    fe = rec.get("first_error") or {}
    key = next((i for i, s in enumerate(traj.steps) if s.get("step") == fe.get("step")), None)
    out = []

    if name == "没买却声称已买":
        buys = [i for i, s in enumerate(traj.steps) if s.get("tool") == "buy_now"]
        k = buys[-1] if buys else n - 1
        page = page_name(traj.state_before(k)) if buys else "（没有点过购买）"
        out += ["**评测器的结论**：这道题其实没有买成。"
                f"agent 在第 {traj.steps[k].get('step')} 步点了“购买”，但它当时停在 **{page}**，那一页没有购买按钮，所以没有生效；"
                "可它最后告诉用户“已经买好了”。", "", "**相关轨迹**：", *step_lines(traj, k - 3, k + 1, gold_asin), "",
                f"**agent 最后说的话**：{words}", "",
                "**请回答**：", f"1. 第 {traj.steps[k].get('step')} 步点购买时，它是不是停在{page}，而且页面显示“动作无效”？（是/否）",
                "2. 它最后是不是跟用户说了已经买好了？（是/否）", "", "两个都是“是”，评测器就判断对了。"]
    elif name == "失败原因是“动作无效”":
        intended = "打开标准答案那件商品" if fe.get("intended") == "open_gold" else "选标准答案要求的规格"
        k = key if key is not None else 0
        out += [f"**评测器的结论**：agent 本来想{intended}，方向是对的；但第 {fe.get('step')} 步做这个动作时，"
                f"它停在**{page_name(traj.state_before(k))}**，当时页面上没有这个选项，动作没有生效，后面也没能补救。", "",
                "**相关轨迹**：", *step_lines(traj, k - 3, k + 3, gold_asin), "",
                "**请回答**：", f"1. 第 {fe.get('step')} 步是不是显示“动作无效”？（是/否）",
                f"2. 它想做的事，确实是{intended}吗？（标准答案商品是 {gold_asin}，规格是 {gold['options']}）（是/否）"]
    elif name in ("买对了，而且过程完整", "买的不是标准答案，但判为可接受"):
        b = traj.buy_index() or n - 1
        head = ("**评测器的结论**：买对了（就是标准答案那件，规格也对），而且购买前该核对的都在页面上看到过。"
                if name.startswith("买对了") else
                "**评测器的结论**：买的不是标准答案那件，但它满足用户所有能核实的关键要求（品类、品牌、型号、规格、预算），所以算“可接受”。")
        purchase = traj.purchase
        out += [head, "", f"**买的是**：{purchase.get('name', '')}，规格 {purchase.get('options')}，{purchase.get('price')} 元", "",
                *constraint_table(rubric, rec), "", "**购买前的最后几步**：", *step_lines(traj, b - 3, b + 1, gold_asin), "",
                "**请回答**：", "1. 表里每条“满足/违反/无法确认”的判断，和依据对得上吗？有没有明显判错的？（有就写出是哪条）"]
        if not name.startswith("买对了"):
            out.append("2. 你觉得这件商品真能满足用户的需求吗？（是/否）")
    elif name == "买对了，但有要求没核对就说满足":
        b = traj.buy_index() or n - 1
        bad = rec.get("unverified", []) + (rec.get("claims") or {}).get("false_verification", [])
        descs = [c["description"] for c in rubric["constraints"] if c["id"] in bad]
        out += [f"**评测器的结论**：买对了，但这些要求它没有真正核对（或核对结果其实不满足），却说满足了：{'、'.join(descs)}。", "",
                *constraint_table(rubric, rec), "", f"**agent 购买前说的话**：{words}", "",
                "**请回答**：", f"1. 对“{'、'.join(descs)}”，评测器说它没核对清楚或其实不满足，你同意吗？（是/否）"]
    elif name == "标准答案本身和需求矛盾":
        selfcheck = {}
        for line in (ROOT / "data" / "cache" / "gold_selfcheck_dev.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["task_id"] == rec["task_id"]:
                selfcheck = row
        cons = {c["id"]: c for c in rubric["constraints"]}
        details = [f"- 用户要求“{cons[r['id']]['description']}”，标准答案：{r.get('reason', '')}"
                   for r in selfcheck.get("results", []) if r["id"] in selfcheck.get("hard_violated", []) + selfcheck.get("hard_unknown", [])
                   and cons[r["id"]]["type"] in ("category", "brand", "model", "spec", "budget")]
        out += ["**评测器的结论**：这道题的标准答案（gold 商品）本身就和用户需求对不上，所以这道题的“对错”不可靠，报告里单独列出。", "",
                f"**标准答案**：{gold['title']}，规格 {gold['options']}", "", *details, "",
                "**请回答**：", "1. 需求和标准答案是不是确实对不上？（是/否）"]
    elif name == "加了插件仍然失败":
        reason = {"search_miss": "从没搜到标准答案那件商品", "missed_candidate": "标准答案出现在搜索结果里，但它没有打开",
                  "wrong_rejection": "打开过标准答案，后来放弃了，买了别的", "wrong_option": "买对了商品，但规格选错了",
                  "missing_option": "买对了商品，但有一项规格没选", "invalid_action": "想做对的事，但动作无效",
                  "false_completion": "没买成却说买了", "early_stop": "没买就停了", "budget_exhausted": "步数用完了"}.get(fe.get("type"), fe.get("type"))
        k = key if key is not None else max(0, n - 3)
        out += [f"**评测器的结论**：失败原因是“{reason}”，" + (f"关键在第 {fe.get('step')} 步。" if fe.get("step") else "（整条轨迹都没搜到）"), "",
                f"**标准答案**：{gold_asin} {short(gold['title'], 30)}，规格 {gold['options']}", "",
                f"**实际买的**：{traj.purchase.get('asin')} {short(traj.purchase.get('name'), 30)}，规格 {traj.purchase.get('options')}" if traj.purchased else "**实际**：没有买", "",
                "**相关轨迹**：", *step_lines(traj, k - 3, k + 2, gold_asin), "",
                "**请回答**：", "1. 评测器说的失败原因对吗？（是/否）"]
    elif name == "LLM 评审和日志对不上":
        calib = rec.get("judge_calibration") or {}
        wrong = [k for k, v in calib.items() if v is False]
        labels = {"purchased": "有没有真正买下", "saw_final_price_before_buy": "购买前有没有看到最终价格",
                  "opened_description": "有没有打开过 Description 详情子页", "attempted_invalid_action": "有没有做过无效动作"}
        obj = (rec.get("judge") or {}).get("objective") or {}
        lines = [f"- {labels.get(k, k)}：LLM 评审说“{'是' if obj.get(k) else '否'}”，规则根据日志判断为“{'否' if obj.get(k) else '是'}”" for k in wrong]
        out += ["**评测器的结论**：LLM 评审在下面这个客观问题上答错了（和日志对不上）。", "", *lines, "",
                "**完整轨迹**：", *step_lines(traj, 0, n, gold_asin), "",
                "**请回答**：", "1. 看轨迹，是 LLM 评审错了，还是规则错了？（评审错 / 规则错）"]
    return out


CATEGORIES = [
    ("没买却声称已买", "off", lambda r: r.get("false_completion_claim"), 2),
    ("失败原因是“动作无效”", "off", lambda r: (r.get("first_error") or {}).get("type") == "invalid_action", 3),
    ("买对了，而且过程完整", "any", lambda r: r.get("success_class") == "process_complete", 4),
    ("买对了，但有要求没核对就说满足", "any", lambda r: r.get("success_class") == "lucky" and not r.get("gold_conflict"), 2),
    ("买的不是标准答案，但判为可接受", "any", lambda r: r.get("acceptable_alternative"), 2),
    ("标准答案本身和需求矛盾", "any", lambda r: r.get("gold_conflict"), 2),
    ("加了插件仍然失败", "on", lambda r: bool(r.get("first_error")), 3),
    ("LLM 评审和日志对不上", "any", lambda r: any(v is False for v in (r.get("judge_calibration") or {}).values()), 2),
]


def load(run_dir: Path) -> list:
    return [json.loads(line) for line in (run_dir / "eval" / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("off", type=Path)
    parser.add_argument("on", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "spotcheck_dev.md")
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()

    gold = {json.loads(l)["task_id"]: json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()}
    rng = random.Random(args.seed)
    pools = {"off": [(args.off, r) for r in load(args.off)], "on": [(args.on, r) for r in load(args.on)]}
    pools["any"] = pools["off"] + pools["on"]
    picked, used = [], set()
    for name, src, pred, k in CATEGORIES:
        cands = [(run, r) for run, r in pools[src] if pred(r) and r["task_id"] not in used]
        for run, r in rng.sample(cands, min(k, len(cands))):
            used.add(r["task_id"])
            picked.append((name, run, r))

    lines = ["# 人工抽查清单（开发集）", "",
             "**怎么做**：评测器对每条轨迹下了结论。每条下面写着它的结论、相关的几步实际经过，以及一两个是/否问题。"
             "读完回答问题，就知道评测器判断得对不对。最后在“你的结论”那行写“对”，或者写哪里不对。", "",
             "**名词**：标准答案 = 出题时对应的那件商品和规格（gold）；“动作无效” = 点了当前页面上没有的按钮或选项，环境不报错，页面也不变，但照样扣一步。", "",
             f"共 {len(picked)} 条，来自原版（{args.off.name}）和加插件（{args.on.name}）两次运行。", ""]
    trace_dir = args.out.parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for idx, (name, run, r) in enumerate(picked, start=1):
        rubric = json.loads((ROOT / "data" / "rubrics" / f"{r['task_id']}.json").read_text(encoding="utf-8"))
        which = "原版" if run == args.off else "加插件"
        lines += [f"## 第 {idx} 条：{name}（{which}，第 {r['task_id']} 题）", "", f"**用户需求**：{rubric['query']}", ""]
        lines += build_item(name, run, r, rubric, gold[r["task_id"]])
        trace = trace_dir / f"{idx:02d}_{run.name}_task_{r['task_id']}.txt"
        full = subprocess.run([sys.executable, str(ROOT / "scripts" / "show_traj.py"), str(run / f"task_{r['task_id']}"), "--eval"],
                              capture_output=True, text=True, check=True).stdout
        trace.write_text(full, encoding="utf-8")
        lines += ["", f"**完整执行轨迹**：[{trace.name}](traces/{trace.name})（每一回合模型说了什么、调用了什么工具、页面返回了什么）", "",
                  "**你的结论**：", "", "---", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {args.out}，共 {len(picked)} 条")


if __name__ == "__main__":
    main()

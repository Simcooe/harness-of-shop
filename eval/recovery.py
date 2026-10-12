#!/usr/bin/env python3
"""多轮实验汇总：把 A（完整需求）、B（模糊需求、不能问）、C（能问）、D（能问 + 提问策略）放在一起比较。

恢复率 = (X − B) / (A − B)：信息缺失造成的损失，靠提问补回了多少。只在信息缺失组上算；对照组单列，用来看过度提问。
四组都要先跑过 eval/evaluate.py。

用法：python3 eval/recovery.py --multiturn data/multiturn/test.jsonl \
        --A runs/test200_v1 --B runs/test_mt_B --C runs/test_mt_C --D runs/test_mt_D
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from compare import load, mcnemar_exact

ARMS = {"A": "完整需求", "B": "模糊、不能问", "C": "模糊、能问", "D": "模糊、能问+策略"}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def table(runs: dict, ids: list, title: str) -> list:
    L = [f"{title}（{len(ids)} 题）", f"  {'组':<18}{'严格成功':>8}{'可接受':>8}{'有购买':>8}{'平均提问':>8}{'环境步数':>8}{'prompt token':>13}"]
    for arm, rows in runs.items():
        sub = [rows[i] for i in ids if i in rows]
        if not sub:
            continue
        n = len(sub)
        nq = mean((r.get("asking") or {}).get("n_questions", 0) for r in sub)
        L.append(f"  {arm} {ARMS[arm]:<14}{100 * mean(r['strict_success'] for r in sub):>7.0f}%{100 * mean(r['acceptable_core'] for r in sub):>7.0f}%"
                 f"{100 * mean(r['purchased'] for r in sub):>7.0f}%{nq:>8.2f}{mean(r['env_steps'] for r in sub):>8.1f}{mean(r['prompt_tokens'] for r in sub):>13.0f}")
    return L


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiturn", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path)
    args = parser.parse_args()

    mt = {json.loads(l)["task_id"]: json.loads(l) for l in args.multiturn.read_text(encoding="utf-8").splitlines() if l.strip()}
    runs = {arm: load(getattr(args, arm)) for arm in ARMS if getattr(args, arm)}
    common = [i for i in mt if all(i in rows for rows in runs.values())]
    missing = [i for i in common if mt[i]["group"] == "missing"]
    control = [i for i in common if mt[i]["group"] == "control"]

    L = table(runs, missing, "信息缺失组") + [""] + table(runs, control, "对照组（需求本来就完整）") + [""]

    if "A" in runs and "B" in runs:
        for key, name in (("acceptable_core", "可接受购买"), ("strict_success", "严格成功")):
            a = mean(runs["A"][i][key] for i in missing)
            b = mean(runs["B"][i][key] for i in missing)
            parts = []
            for arm in ("C", "D"):
                if arm in runs:
                    x = mean(runs[arm][i][key] for i in missing)
                    parts.append(f"{arm} {100 * (x - b) / (a - b):.0f}%" if a != b else f"{arm} 无法计算")
            L.append(f"恢复率（{name}，A {100 * a:.0f}% / B {100 * b:.0f}%）：{'，'.join(parts)}")

    L.append("配对比较（信息缺失组，可接受购买）：")
    for x, y in (("B", "C"), ("C", "D"), ("B", "D"), ("D", "A")):
        if x in runs and y in runs:
            up = sum(1 for i in missing if not runs[x][i]["acceptable_core"] and runs[y][i]["acceptable_core"])
            down = sum(1 for i in missing if runs[x][i]["acceptable_core"] and not runs[y][i]["acceptable_core"])
            L.append(f"  {x} → {y}：变好 {up} 题，变差 {down} 题，McNemar p = {mcnemar_exact(down, up):.2g}")

    for arm in ("C", "D"):
        if arm not in runs:
            continue
        rows = runs[arm]
        sub = [rows[i] for i in missing if rows[i].get("asking")]
        qs = [q for r in sub for q in r["asking"]["questions"]]
        kinds = Counter(q["kind"] for q in qs)
        hidden = sum(len(r["asking"]["hidden"]) for r in sub)
        got = sum(len(r["asking"]["hidden_asked"]) for r in sub)
        fail = Counter(r["asking"]["ask_failure"] for r in sub if r["asking"]["ask_failure"])
        ctrl = [rows[i] for i in control if rows[i].get("asking")]
        L.append(f"{arm} 提问：信息缺失组提问率 {100 * mean(r['asking']['n_questions'] > 0 for r in sub):.0f}%，被删约束问到 {got}/{hidden}；"
                 f"问题 有效 {kinds.get('effective', 0)} / 冗余 {kinds.get('redundant', 0)} / 不必要 {kinds.get('unnecessary', 0)}；"
                 f"该问没问 {fail.get('should_have_asked', 0)} 题，问了没用 {fail.get('asked_not_used', 0)} 题；"
                 f"对照组提问率 {100 * mean(r['asking']['n_questions'] > 0 for r in ctrl):.0f}%")
    print("\n".join(L))


if __name__ == "__main__":
    main()

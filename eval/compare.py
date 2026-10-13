#!/usr/bin/env python3
"""配对比较两次运行（同一批题）：成败翻转表、McNemar 精确检验、bootstrap 置信区间、插件干预统计。

用法：python3 eval/compare.py runs/<基线> runs/<实验>   （两边都要先跑过 eval/evaluate.py）
"""
import argparse
import json
import random
from collections import Counter
from math import comb
from pathlib import Path


def load(run_dir: Path) -> dict:
    path = run_dir / "eval" / "tasks.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["task_id"]: r for r in rows if not r["infra_error"]}


def mcnemar_exact(b: int, c: int) -> float:
    """b、c 为两个方向的翻转数；双侧精确二项检验 p 值。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def bootstrap_diff(pairs: list, iters: int = 10000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(iters):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(b - a for a, b in sample) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def interventions(run_dir: Path) -> Counter:
    counts = Counter()
    for f in run_dir.glob("task_*/harness_events.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                counts[json.loads(line)["type"]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("exp", type=Path)
    args = parser.parse_args()

    base, exp = load(args.base), load(args.exp)
    ids = sorted(set(base) & set(exp))
    pairs = [(int(base[i]["strict_success"]), int(exp[i]["strict_success"])) for i in ids]
    b = sum(1 for x, y in pairs if x == 1 and y == 0)
    c = sum(1 for x, y in pairs if x == 0 and y == 1)
    n = len(ids)
    sa, sb = sum(x for x, _ in pairs), sum(y for _, y in pairs)
    lo, hi = bootstrap_diff(pairs)

    print(f"配对题数 {n}：{args.base.name} vs {args.exp.name}")
    print(f"严格成功  {sa}/{n} → {sb}/{n}（差 {100 * (sb - sa) / n:+.1f} 个百分点，95% bootstrap CI [{100 * lo:+.1f}, {100 * hi:+.1f}]）")
    print(f"翻转      失败→成功 {c} 题，成功→失败 {b} 题，McNemar 精确检验 p = {mcnemar_exact(b, c):.2g}")
    if b:
        print(f"  成功→失败的题：{[i for i in ids if base[i]['strict_success'] and not exp[i]['strict_success']]}")

    def rate(rows, key):
        return sum(1 for i in ids if rows[i].get(key)) / n

    def stage(rows, key):
        return sum(1 for i in ids if rows[i][key] is not None) / n

    print("\n指标                      基线     实验")
    for label, fn in [
        ("严格成功", lambda r: rate(r, "strict_success")),
        ("可接受购买（core）", lambda r: rate(r, "acceptable_core")),
        ("有购买", lambda r: rate(r, "purchased")),
        ("没买却声称已买", lambda r: rate(r, "false_completion_claim")),
        ("幸运成功", lambda r: sum(1 for i in ids if r[i].get("success_class") == "lucky") / n),
        ("gold 出现在搜索结果", lambda r: stage(r, "retrieved_at")),
        ("打开了 gold 商品", lambda r: stage(r, "opened_at")),
        ("gold 规格全部选对", lambda r: stage(r, "options_correct_at")),
    ]:
        print(f"  {label:<20}{100 * fn(base):6.0f}%  {100 * fn(exp):6.0f}%")
    for label, key in [("平均环境步数", "env_steps"), ("平均 LLM 回合", "llm_turns"), ("平均 prompt token", "prompt_tokens"), ("平均输出 token", "output_tokens")]:
        va = sum(base[i][key] for i in ids) / n
        vb = sum(exp[i][key] for i in ids) / n
        print(f"  {label:<20}{va:8.1f} {vb:8.1f}")
    derived = [
        ("平均提问次数", lambda r: (r.get("asking") or {}).get("n_questions", 0)),
        ("没命中缓存的 prompt", lambda r: r["prompt_tokens"] - r["cached_tokens"]),
        ("维护状态 token", lambda r: r.get("state_prompt_tokens", 0) + r.get("state_output_tokens", 0)),
    ]
    for label, fn in derived:
        va = sum(fn(base[i]) for i in ids) / n
        vb = sum(fn(exp[i]) for i in ids) / n
        print(f"  {label:<20}{va:8.1f} {vb:8.1f}")

    def invalid_share(rows):
        total = Counter()
        for i in ids:
            total.update(rows[i]["label_counts"])
        steps = sum(total.values())
        return total["invalid"] / steps if steps else 0.0

    print(f"  {'无效动作占比':<20}{100 * invalid_share(base):6.0f}%  {100 * invalid_share(exp):6.0f}%")
    if all(base[i].get("judge") and exp[i].get("judge") for i in ids):
        print("\nLLM 评审（作者的 5 个维度，平均分 0–2）")
        for d in ("search_strategy", "candidate_utilization", "evidence_verification", "decision_quality", "termination_efficiency"):
            va = [base[i]["judge"]["scores"].get(d) for i in ids if base[i]["judge"]["scores"].get(d) is not None]
            vb = [exp[i]["judge"]["scores"].get(d) for i in ids if exp[i]["judge"]["scores"].get(d) is not None]
            print(f"  {d:<24}{sum(va) / len(va):6.2f}  {sum(vb) / len(vb):6.2f}")
    for name, rows in (("基线", base), ("实验", exp)):
        fe = Counter(rows[i]["first_error"]["type"] for i in ids if rows[i].get("first_error"))
        print(f"  首错类型（{name}）{dict(fe.most_common())}")
    print(f"\n插件干预次数（{args.exp.name}）：{dict(interventions(args.exp).most_common())}")


if __name__ == "__main__":
    main()

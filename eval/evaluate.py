#!/usr/bin/env python3
"""过程评测：读一次运行的日志，输出 6 个面板（结果、过程正确性、失败诊断、过程质量、harness 干预、效率与成本）；
多轮运行另加第 7 个面板（提问）。

用法：python3 eval/evaluate.py runs/<run> [--judge] [--diagnose] [--workers 8]

只读日志、冻结的需求约束（data/rubrics/）和离线 gold（data/private/gold.jsonl），不调用环境，不用环境奖励。
LLM 调用（约束核对、自述抽取、过程评审）都有缓存，重复运行不会重复请求。
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asking import analyze as analyze_asking, load_multiturn, paper_multiturn  # noqa: E402
from checker import load_products, load_rubric  # noqa: E402
from consent import consent  # noqa: E402
from diagnose import diagnose  # noqa: E402
from judge import DIMENSIONS, judge  # noqa: E402
from process import buy_blocks, claims, first_error, interventions, outcome, verification  # noqa: E402
from state_check import analyze as analyze_state, card_specs  # noqa: E402
from traj import Trajectory, load_jsonl, norm, options_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "private" / "gold.jsonl"
ACTION_RE = re.compile(r"^(search|click)\[(.*)\]$", re.S)
CLAIM_RE = re.compile(r"购买已完成|完成购买|已.{0,4}购买|成功下单|已.{0,2}下单|购买成功|completed?\s+(?:your|the)\s+purchase|purchase\s+(?:has been|was|is)\s+(?:completed|submitted)", re.I)
STAGE_NAMES = {
    "search_miss": "检索失败", "missed_candidate": "漏看", "wrong_rejection": "误拒", "wrong_option": "规格错选",
    "missing_option": "规格漏选", "false_completion": "假完成", "early_stop": "提前放弃", "budget_exhausted": "步数耗尽",
    "invalid_action": "无效动作（想打开 gold 或选 gold 规格，但动作在当前页无效）", "other": "其他",
}


def load_gold() -> dict:
    return {row["task_id"]: row for row in load_jsonl(GOLD_PATH)}


def load_gold_conflicts() -> set:
    """gold 自检没通过（gold 本身与需求矛盾或评测器误判）的题，报告里单列。"""
    conflicts = set()
    for path in (ROOT / "data" / "cache").glob("gold_selfcheck_*.jsonl"):
        for row in load_jsonl(path):
            if not row.get("acceptable_core"):
                conflicts.add(row["task_id"])
    return conflicts


def label_step(action: str, prev_state: dict, seen: dict) -> str:
    """给一步环境动作打标签。无效点击：目标不在上一页的可点击列表里，环境原地不动但计步。"""
    m = ACTION_RE.match(action.strip())
    if not m:
        return "malformed"
    kind, arg = m.group(1), m.group(2)
    if kind == "search":
        q = norm(arg)
        label = "search_repeat" if q in seen["queries"] else "search_new"
        seen["queries"].add(q)
        return label
    clickables = {norm(a) for a in (prev_state or {}).get("actions", [])}
    target = norm(arg)
    if target not in clickables:
        return "invalid"
    prev_page = (prev_state or {}).get("page_type")
    if target == "buynow":
        return "buy"
    if target in {"backtosearch", "<prev", "next>"}:
        return "nav"
    if target in {"description", "features", "reviews", "attributes"}:
        return "view_subpage"
    if prev_page == "search_results":
        label = "open_repeat" if target in seen["opened"] else "open_new"
        seen["opened"].add(target)
        return label
    if prev_page in {"product_detail", "information_subpage"}:
        return "select_option"
    return "other_click"


def failure_stage(rec: dict):
    if rec["strict_success"]:
        return None
    if rec["retrieved_at"] is None:
        return "search"
    if rec["opened_at"] is None:
        return "pick_product"
    if rec["options_correct_at"] is None:
        return "pick_options"
    if not rec["bought_gold"]:
        return "commit"
    return "options_at_purchase"


def evaluate_task(task_dir: Path, gold: dict, products: dict, conflicts: set, use_judge: bool, multiturn: dict, use_diagnose: bool = False) -> dict:
    traj = Trajectory(task_dir)
    g = gold[traj.task_id]
    gold_asin = str(g["asin"])
    rec = {"task_id": traj.task_id, "seconds": traj.done.get("seconds")}
    rc = traj.done.get("returncode")
    # 没有 done.json 说明这题没跑完（比如环境槽位占满、reset 一直失败），算基础设施异常，不当成没买
    rec["infra_error"] = traj.done.get("error") or (rc not in (0, None) and f"rc={rc}") or (None if traj.done else "没有 done.json，没跑完")
    if any(e.get("type") == "env_error" for e in traj.harness):
        rec["infra_error"] = rec["infra_error"] or "env_error"

    # 里程碑和步骤标签
    seen = {"queries": set(), "opened": set()}
    labels, retrieved_at, opened_at, options_at = [], None, None, None
    prev_state = traj.initial_state
    for i, s in enumerate(traj.steps, start=1):
        result = s.get("result") or {}
        state = result.get("observation_state") or {}
        labels.append(label_step(s.get("action", ""), prev_state, seen))
        page = state.get("page_type")
        if page == "search_results" and retrieved_at is None and any(str(p.get("asin")) == gold_asin for p in state.get("products") or []):
            retrieved_at = i
        if page in ("product_detail", "information_subpage") and str((state.get("product") or {}).get("asin")) == gold_asin:
            opened_at = opened_at or i
            if options_at is None and options_match(state.get("selected_options"), g["options"]):
                options_at = i
        if not result.get("done"):
            prev_state = state

    purchase = traj.purchase
    bought_gold = traj.purchased and str(purchase.get("asin")) == gold_asin
    rec.update({
        "env_steps": len(traj.steps),
        "termination": (traj.terminal or {}).get("termination_reason") or "agent_stopped",
        "purchased": traj.purchased,
        "bought_gold": bought_gold,
        "strict_success": bought_gold and options_match(purchase.get("options"), g["options"]),
        "env_gold_purchase": (traj.terminal or {}).get("termination_reason") == "gold_purchase",
        "retrieved_at": retrieved_at, "opened_at": opened_at, "options_correct_at": options_at,
        "labels": labels, "label_counts": dict(Counter(labels)),
        "serial_blocks": sum(1 for e in traj.harness if e.get("type") == "serial_block"),
        "gold_conflict": traj.task_id in conflicts,
    })
    usage = Counter()
    for u in [a["usage"] for a in traj.assistant] + traj.buyer_usage:
        usage["prompt"] += u.get("input", 0) + u.get("cacheRead", 0)
        usage["cached"] += u.get("cacheRead", 0)
        usage["output"] += u.get("output", 0)
    rec.update({"llm_turns": len(traj.assistant) + len(traj.buyer_usage), "prompt_tokens": usage["prompt"], "cached_tokens": usage["cached"], "output_tokens": usage["output"]})
    # 回合：对外的动作（商店动作和对用户说的话）；分工模式下导购和买手内部的调用不算，单独记调用次数
    rec.update({"protocol_turns": traj.turns_used, "roles": traj.roles, "buyer_llm_calls": len(traj.buyer_usage)})
    # 任务状态层维护两张表时调用小模型的用量（agent 自己的调用不含这部分）
    state = Counter()
    for r in load_jsonl(task_dir / "state_log.jsonl"):
        u = r.get("usage") or {}
        if not u:
            continue
        state["calls"] += 1
        state["prompt"] += u.get("prompt_tokens", 0)
        state["cached"] += u.get("prompt_cache_hit_tokens", (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        state["output"] += u.get("completion_tokens", 0)
    rec.update({"state_calls": state["calls"], "state_prompt_tokens": state["prompt"], "state_cached_tokens": state["cached"], "state_output_tokens": state["output"]})
    last_text = traj.assistant[-1]["text"] if traj.assistant else ""
    rec["false_completion_claim_regex"] = not traj.purchased and bool(CLAIM_RE.search(last_text))
    rec["failure_stage"] = failure_stage(rec)

    rubric = load_rubric(traj.task_id)
    if rubric is None:
        rec["eval_error"] = "缺少需求约束，先运行 eval/rubric.py"
        return rec

    # 1. 结果：需求满足度
    out = outcome(traj, rubric, products)
    rec["check"] = out["results"]
    for k in ("acceptable", "acceptable_core", "acceptable_lenient", "hard_violated", "hard_unknown", "violated_types"):
        rec[k] = out[k]
    rec["acceptable_alternative"] = rec["acceptable_core"] and not bought_gold
    # 下单前用户有没有点头（论文多轮设置）
    if traj.protocol == "paper":
        rec["consent"] = consent(traj)

    # 2. 过程正确性：购买前核验 + 自述核对
    ver = verification(traj, rubric, out["results"])
    rec["verification"] = ver["statuses"]
    rec["unverified"] = ver["unverified"]
    rec["knowingly_violated"] = ver["knowingly_violated"]
    rec["price_seen"] = ver["price_seen"]
    cl = claims(traj, rubric, out["results"], ver["statuses"])
    rec["claims"] = {k: v for k, v in cl.items() if k != "words"}
    rec["false_completion_claim"] = cl["false_purchase_claim"]
    if rec["strict_success"]:
        lucky = bool(ver["unverified"] or cl["false_verification"])
        rec["success_class"] = "lucky" if lucky else "process_complete"
    else:
        rec["success_class"] = None

    # 3. 失败诊断：所有没有严格成功的题都做首错定位。买了可接受替代品的题也算，
    #    否则"商品买对了、有规格没点"这类失败会因为买到的商品可以接受而被漏掉
    if not rec["strict_success"]:
        rec["first_error"] = first_error(traj, g, cl["false_purchase_claim"])
    else:
        rec["first_error"] = None
    # 失败原因：首错定位只说得出错在哪一步，这里判断为什么错
    if use_diagnose and not rec["strict_success"]:
        rec["diagnosis"] = diagnose(traj, rubric, g, rec, products)

    # 5. harness 干预；拦下的购买放行的话会不会严格成功
    rec["interventions"] = interventions(traj)
    rec["buy_blocks"] = buy_blocks(traj, g)

    # 7. 多轮提问
    if traj.protocol == "paper":
        mt = paper_multiturn(rubric, traj.agent_query)
        rec["asking"] = analyze_asking(traj, rubric, mt, out["results"], rec["acceptable_core"], check_answers=False)
        rec["asking"]["stated"] = mt["stated"]
    elif traj.multiturn and traj.task_id in multiturn:
        rec["asking"] = analyze_asking(traj, rubric, multiturn[traj.task_id], out["results"], rec["acceptable_core"])

    # 8. 任务状态层的需求表准不准
    state_table = analyze_state(traj, rubric)
    if state_table:
        rec["state_table"] = state_table
    specs = card_specs(traj, g)
    if specs:
        rec["state_specs"] = specs

    # 4. 过程质量（LLM 评审）与校准
    if use_judge:
        j = judge(traj, rubric)
        rec["judge"] = j
        truth = {
            "purchased": traj.purchased,
            "saw_final_price_before_buy": ver["price_seen"] if traj.purchased else None,
            "opened_description": any(
                ((s.get("result") or {}).get("observation_state") or {}).get("subpage", "").lower() == "description" for s in traj.steps
            ),
            "attempted_invalid_action": rec["label_counts"].get("invalid", 0) > 0 or any(e.get("type") == "guard_block" for e in traj.harness),
        }
        rec["judge_calibration"] = {
            k: (None if truth[k] is None or j["objective"].get(k) is None else bool(j["objective"].get(k)) == truth[k]) for k in truth
        }
    return rec


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def summarize_asking(recs: list) -> list:
    L = ["【7 多轮提问】"]
    groups = {"missing": "信息缺失组", "control": "对照组（需求本来就完整）", "paper": "论文多轮设置（开场只有模糊方向）"}
    for g, name in groups.items():
        sub = [r for r in recs if r["asking"]["group"] == g]
        if not sub:
            continue
        n = len(sub)
        nq = [r["asking"]["n_questions"] for r in sub]
        L.append(f"  {name} {n} 题：严格成功 {pct(sum(r['strict_success'] for r in sub), n)}，可接受购买 {pct(sum(r['acceptable_core'] for r in sub), n)}；"
                 f"提问率 {pct(sum(1 for x in nq if x), n)}，平均提问 {mean(nq):.2f} 次")
        if g in ("missing", "paper"):
            what = "被删的约束" if g == "missing" else "开场没说的约束"
            hidden = sum(len(r["asking"]["hidden"]) for r in sub)
            got = sum(len(r["asking"]["hidden_asked"]) for r in sub)
            bought = [r for r in sub if r["purchased"]]
            asked_b = sum(len(r["asking"]["hidden_asked"]) for r in bought)
            used = sum(len(r["asking"]["hidden_asked_used"]) for r in bought)
            if g == "paper":
                stated = sum(len(r["asking"].get("stated") or []) for r in sub)
                L.append(f"    开场白说出的约束 {pct(stated, stated + hidden)}")
            L.append(f"    {what}被问到 {pct(got, hidden)}；问到的约束在最终购买里满足 {pct(used, asked_b)}（答案利用率）")
            fail = Counter(r["asking"]["ask_failure"] for r in sub if r["asking"]["ask_failure"])
            L.append(f"    不可接受、且{what}没满足的题：该问没问 {fail.get('should_have_asked', 0)}，问了没用 {fail.get('asked_not_used', 0)}")
    qs = [q for r in recs for q in r["asking"]["questions"]]
    if qs:
        kinds = Counter(q["kind"] for q in qs)
        L.append(f"  问题分类（共 {len(qs)} 个）：有效 {kinds.get('effective', 0)}，冗余 {kinds.get('redundant', 0)}，不必要 {kinds.get('unnecessary', 0)}")
        chk = [q["answer_check"] for q in qs if q["answer_check"] is not None]
        if chk:
            L.append(f"  模拟买家：回答针对问题 {pct(sum(c.get('relevant') is True for c in chk), len(chk))}，"
                     f"忠于需求 {pct(sum(c.get('faithful') is True for c in chk), len(chk))}，"
                     f"泄露没问到的需求 {pct(sum(c.get('leaked') is True for c in chk), len(chk))}")
    return L


def summarize(recs: list) -> str:
    valid = [r for r in recs if not r["infra_error"] and not r.get("eval_error")]
    n = len(valid)
    clean = [r for r in valid if not r["gold_conflict"]]
    succ = [r for r in valid if r["strict_success"]]
    L = [f"有效题数 {n}（基础设施或评测异常 {len(recs) - n}；gold 自检未通过、标注可能矛盾的题 {n - len(clean)}）", ""]

    L.append("【1 结果】")
    L.append(f"  严格成功（买到 gold 且规格全对）  {pct(len(succ), n)}；去掉标注矛盾题 {pct(sum(r['strict_success'] for r in clean), len(clean))}")
    L.append(f"  可接受购买（core 口径）            {pct(sum(r['acceptable_core'] for r in valid), n)}"
             f"   [strict 口径 {pct(sum(r['acceptable'] for r in valid), n)}，lenient 口径 {pct(sum(r['acceptable_lenient'] for r in valid), n)}]")
    L.append(f"  其中买的不是 gold 但可接受          {pct(sum(r['acceptable_alternative'] for r in valid), n)}")
    L.append(f"  有购买 {pct(sum(r['purchased'] for r in valid), n)}；结束方式 {dict(Counter(r['termination'] for r in valid))}")
    vt = Counter(t for r in valid if r["purchased"] for t in r["violated_types"])
    L.append(f"  购买中被违反的 hard 约束类型       {dict(vt.most_common())}")
    consented = [r for r in valid if r.get("consent")]
    if consented:
        att = Counter(r["consent"]["attitude"] for r in consented)
        names = {"agree": "明确同意", "none": "没表态（还在补充需求或提问）", "refuse": "刚说完不买", "noask": "没问过用户"}
        L.append(f"  下单前用户的最后一句话（{len(consented)} 次下单）：" + "，".join(f"{names[k]} {att.get(k, 0)}" for k in names))
        agreed_right = sum(1 for r in consented if r["strict_success"] and r["consent"]["attitude"] == "agree")
        L.append(f"  用户点头后买对（严格成功，且下单前用户最后一句话明确同意） {pct(agreed_right, n)}")

    L.append("【2 过程正确性】")
    sc = Counter(r["success_class"] for r in succ)
    L.append(f"  成功中过程完备 {sc.get('process_complete', 0)}，幸运成功 {sc.get('lucky', 0)}（有未核验约束或虚假核验声明）")
    unv = Counter()
    for r in valid:
        rub = load_rubric(r["task_id"])
        types = {c["id"]: c["type"] for c in rub["constraints"]} if rub else {}
        unv.update(types.get(cid, "?") for cid in r["unverified"])
    L.append(f"  购买前未核验的约束（按类型）       {dict(unv.most_common())}")
    L.append(f"  明知违反仍购买的题                 {pct(sum(bool(r['knowingly_violated']) for r in valid), n)}")
    L.append(f"  没买却声称已买                     {pct(sum(r['false_completion_claim'] for r in valid), n)}")
    L.append(f"  声称满足、实际违反或未核验（虚假核验）的题 {pct(sum(bool(r['claims']['false_verification']) for r in valid), n)}")
    L.append(f"  声称的价格与实际成交价不符的题     {pct(sum(r['claims']['price_mismatch'] for r in valid), n)}")

    L.append("【3 失败诊断】")
    fe = Counter(STAGE_NAMES.get(r["first_error"]["type"], r["first_error"]["type"]) for r in valid if r["first_error"])
    L.append(f"  没有严格成功 {sum(1 for r in valid if r['first_error'])} 题，首错类型 {dict(fe.most_common())}")
    bad = Counter(STAGE_NAMES.get(r["first_error"]["type"], r["first_error"]["type"]) for r in valid if r["first_error"] and not r["acceptable_core"])
    L.append(f"    其中买的东西不可接受的 {sum(bad.values())} 题，首错类型 {dict(bad.most_common())}")
    diagnosed = [r for r in valid if r.get("diagnosis")]
    if diagnosed:
        causes = Counter(r["diagnosis"]["cause"] for r in diagnosed)
        tags = Counter(t for r in diagnosed for t in r["diagnosis"].get("tags") or [])
        L.append(f"  失败原因（{len(diagnosed)} 题，按核对过的事实由规则判定）：{dict(causes.most_common())}")
        L.append(f"    附加标签：{dict(tags.most_common())}")
    L.append(f"  漏斗：gold 被搜到 {pct(sum(r['retrieved_at'] is not None for r in valid), n)} → 被打开 {pct(sum(r['opened_at'] is not None for r in valid), n)}"
             f" → 规格全对 {pct(sum(r['options_correct_at'] is not None for r in valid), n)} → 买下 {pct(len(succ), n)}")

    judged = [r for r in valid if r.get("judge")]
    if judged:
        L.append("【4 过程质量（LLM 评审，作者的 5 个维度，0–2 分）】")
        for d in DIMENSIONS:
            dist = Counter(r["judge"]["scores"].get(d) for r in judged)
            L.append(f"  {d:<24} 平均 {mean(r['judge']['scores'].get(d) for r in judged):.2f}  分布 {dict(sorted(dist.items(), key=lambda kv: str(kv[0])))}")
        L.append("  评审校准（评审对客观问题的回答与日志是否一致）：")
        for k in ("purchased", "saw_final_price_before_buy", "opened_description", "attempted_invalid_action"):
            vals = [r["judge_calibration"][k] for r in judged if r["judge_calibration"][k] is not None]
            L.append(f"    {k:<28} 准确率 {pct(sum(vals), len(vals))}")

    L.append("【5 harness 干预】")
    agg = {}
    for r in valid:
        for kind, st in r["interventions"].items():
            a = agg.setdefault(kind, Counter())
            a["count"] += st.get("count", 0)
            a["tasks"] += 1
            a["repeated"] += st.get("repeated", 0)
            a["bought_after"] += int(bool(st.get("bought_after")))
    if agg:
        for kind, a in sorted(agg.items()):
            extra = f"，下一步原样重复被拦动作 {a['repeated']} 次" if kind == "guard_block" else ""
            extra += f"，提醒后最终完成购买的题 {a['bought_after']}/{a['tasks']}" if kind == "finish_hint" else ""
            L.append(f"  {kind:<14} {a['count']} 次，涉及 {a['tasks']} 题{extra}")
    else:
        L.append("  无（插件未加载）")
    blocked = [r for r in valid if r.get("buy_blocks")]
    if blocked:
        right = [r for r in blocked if any(b["was_correct"] for b in r["buy_blocks"])]
        wrong = [r for r in blocked if not any(b["was_correct"] for b in r["buy_blocks"])]
        n_right = sum(b["was_correct"] for r in blocked for b in r["buy_blocks"])
        L.append(f"  拦下购买 {sum(len(r['buy_blocks']) for r in blocked)} 次（{len(blocked)} 题），其中放行就会严格成功的 {n_right} 次")
        L.append(f"    拦下过正确购买的 {len(right)} 题：最后仍严格成功 {sum(r['strict_success'] for r in right)}，因此没买成 {sum(not r['strict_success'] for r in right)}")
        L.append(f"    只拦下过错误购买的 {len(wrong)} 题：最后改成严格成功 {sum(r['strict_success'] for r in wrong)}")

    L.append("【6 效率与成本】")
    total = Counter()
    for r in valid:
        total.update(r["label_counts"])
    steps = sum(total.values())
    wasted = total["invalid"] + total["search_repeat"] + total["open_repeat"] + total["view_subpage"]
    L.append(f"  平均环境步数 {mean(r['env_steps'] for r in valid):.1f}，平均 LLM 回合 {mean(r['llm_turns'] for r in valid):.1f}，平均耗时 {mean(r['seconds'] for r in valid):.1f}s")
    if any(r.get("roles") for r in valid):
        L.append(f"  分工模式：平均用掉的回合（商店动作和对用户说的话） {mean(r['protocol_turns'] for r in valid):.1f}；"
                 f"LLM 调用里导购 {mean(r['llm_turns'] - r['buyer_llm_calls'] for r in valid):.1f} 次、买手 {mean(r['buyer_llm_calls'] for r in valid):.1f} 次")
    L.append(f"  浪费的步数 {pct(wasted, steps)}（无效动作 {total['invalid']}、重复搜索 {total['search_repeat']}、重复打开 {total['open_repeat']}、查看无内容的详情子页 {total['view_subpage']}）")
    L.append(f"  平均 prompt token {mean(r['prompt_tokens'] for r in valid):.0f}（缓存命中 {mean(r['cached_tokens'] for r in valid):.0f}，"
             f"没命中 {mean(r['prompt_tokens'] - r['cached_tokens'] for r in valid):.0f}），平均输出 token {mean(r['output_tokens'] for r in valid):.0f}")
    if any(r.get("state_calls") for r in valid):
        L.append(f"  维护任务状态（小模型）：平均每题调用 {mean(r['state_calls'] for r in valid):.1f} 次，prompt token {mean(r['state_prompt_tokens'] for r in valid):.0f}"
                 f"（缓存命中 {mean(r['state_cached_tokens'] for r in valid):.0f}），输出 token {mean(r['state_output_tokens'] for r in valid):.0f}")
    asked = [r for r in valid if r.get("asking")]
    if asked:
        L += summarize_asking(asked)
    tabled = [r for r in valid if r.get("state_table")]
    if tabled:
        said = sum(len(r["state_table"]["said"]) for r in tabled)
        kept = sum(len(r["state_table"]["said_in_table"]) for r in tabled)
        rows = sum(r["state_table"]["n_rows"] for r in tabled)
        L.append("【8 任务状态：需求表准不准（每题最后一版）】")
        L.append(f"  用户说出口的约束进了需求表 {pct(kept, said)}；平均每题 {rows / len(tabled):.1f} 行")
        specs = [s for r in valid for s in r.get("state_specs") or []]
        if specs:
            L.append(f"  商品表里正确商品的规格，标的“用户要”和正确规格一致 {pct(sum(s['correct'] for s in specs), len(specs))}")
        L.append(f"  表里的行对得上需求原文的约束 {pct(sum(r['state_table']['rows_matched'] for r in tabled), rows)}，"
                 f"记错的 {pct(sum(r['state_table']['rows_wrong'] for r in tabled), rows)}（其余是用户随口说的、约束清单里没有的）")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--judge", action="store_true", help="同时运行 LLM 过程评审（V4 Pro）")
    parser.add_argument("--diagnose", action="store_true", help="没有严格成功的题，判断失败的根本原因（规则判定，用户说过什么、规格能不能推出来交给模型核对）")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    gold, products, conflicts, multiturn = load_gold(), load_products(), load_gold_conflicts(), load_multiturn()
    task_dirs = sorted(p for p in args.run_dir.glob("task_*") if p.is_dir())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        recs = list(pool.map(lambda d: evaluate_task(d, gold, products, conflicts, args.judge, multiturn, args.diagnose), task_dirs))
    out = args.out or args.run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(recs)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

"""过程层：需求满足度、购买前核验、自述一致性、首错定位、插件干预。输入是一条轨迹和该题冻结的约束。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker import check_purchase, summarize_check  # noqa: E402
from llm import chat_json  # noqa: E402
from traj import Trajectory, gold_value_matches, norm, options_match, squash, visible_text_about  # noqa: E402

CLAIM_MODEL = "deepseek-v4-flash"
SHOP_PAGES = ("product_detail", "information_subpage")

CLAIM_SYSTEM = """你是对话审核员。下面是购物 agent 在任务结束前说的话，以及用户需求拆出的约束列表。只根据 agent 的原话提取它的声明，输出 JSON。

字段：
- claims_purchased：agent 是否声称已经完成购买或下单（只是打算购买、正在购买不算）
- claimed_product：agent 声称购买或推荐的商品名称，没有就留空
- claimed_price：agent 声称的价格（数字），没有就填 null
- claimed_options：agent 声称选定的规格值列表
- constraint_claims：对每条约束，agent 是否声称满足（satisfied）、声称不满足（violated）或没提（not_mentioned），并在 quote 里原样摘录 agent 的相关原话

输出格式：{"claims_purchased": false, "claimed_product": "", "claimed_price": null, "claimed_options": [], "constraint_claims": [{"id": "c1", "claim": "satisfied|violated|not_mentioned", "quote": ""}]}"""


def outcome(traj: Trajectory, rubric: dict, products: dict) -> dict:
    if not traj.purchased:
        return {"results": [], "acceptable": False, "acceptable_core": False, "acceptable_lenient": False,
                "hard_violated": [], "hard_unknown": [], "violated_types": []}
    p = traj.purchase
    product = products.get(str(p["asin"]))
    if product is None:
        return {"results": [], "acceptable": False, "acceptable_core": False, "acceptable_lenient": False,
                "hard_violated": [], "hard_unknown": [], "violated_types": [], "error": "商品库中找不到所购商品"}
    results = check_purchase(rubric, product, p.get("options") or {}, p.get("price"))
    return {"results": results, **summarize_check(rubric, results)}


def verification(traj: Trajectory, rubric: dict, check_results: list) -> dict:
    """每条 hard 约束在购买前的核验状态：依据是核对器给出的证据原文有没有出现在 agent 购买前看过的页面上。"""
    b = traj.buy_index()
    if b is None or not check_results:
        return {"statuses": {}, "unverified": [], "knowingly_violated": [], "price_seen": None}
    asin = str(traj.purchase["asin"])
    price = traj.purchase.get("price")
    states = [traj.initial_state] + [(s.get("result") or {}).get("observation_state") or {} for s in traj.steps[:b]]
    seen = squash("\n".join(visible_text_about(st, asin) for st in states))
    price_seen = price is not None and any(
        st.get("page_type") in SHOP_PAGES and str((st.get("product") or {}).get("asin")) == asin
        and st.get("selected_price") is not None and abs(float(st["selected_price"]) - float(price)) < 0.01
        for st in states
    )
    # 多个规格项都影响价格时，环境自己算不出规格价，页面从不显示，这种情况不算 agent 未核验
    price_shown_ever = any(
        st.get("page_type") in SHOP_PAGES and str((st.get("product") or {}).get("asin")) == asin and st.get("selected_price") is not None
        for st in states
    )
    by_id = {r["id"]: r for r in check_results}
    statuses = {}
    for c in rubric["constraints"]:
        if c["hardness"] != "hard" or c["type"] == "service":
            continue
        r = by_id.get(c["id"], {})
        verdict = r.get("verdict", "unknown")
        if verdict == "unknown":
            statuses[c["id"]] = "no_evidence"
            continue
        if c["type"] == "budget" and not price_seen and not price_shown_ever:
            statuses[c["id"]] = "not_displayed"
            continue
        was_seen = price_seen if c["type"] == "budget" else bool(squash(r.get("evidence_quote", ""))) and squash(r.get("evidence_quote", "")) in seen
        if verdict == "satisfied":
            statuses[c["id"]] = "verified" if was_seen else "unverified"
        else:
            statuses[c["id"]] = "knowingly_violated" if was_seen else "blind_violation"
    return {
        "statuses": statuses,
        "unverified": [k for k, v in statuses.items() if v == "unverified"],
        "knowingly_violated": [k for k, v in statuses.items() if v == "knowingly_violated"],
        "price_seen": price_seen,
    }


def final_words(traj: Trajectory) -> str:
    turn = traj.buy_turn()
    if turn is not None:
        msgs = [a for a in traj.assistant if a["turn"] in (turn - 1, turn)]
    else:
        msgs = traj.assistant[-2:]
    return "\n\n".join(a["text"][:2000] for a in msgs if a["text"])


def claims(traj: Trajectory, rubric: dict, check_results: list, statuses: dict) -> dict:
    words = final_words(traj)
    empty = {"claims_purchased": False, "false_purchase_claim": False, "price_mismatch": False,
             "false_verification": [], "unsupported_claims": [], "words": words}
    if not words:
        return empty
    cons_text = "\n".join(f"{c['id']} {c['description']}" for c in rubric["constraints"])
    raw = chat_json(CLAIM_MODEL, CLAIM_SYSTEM, f"需求约束：\n{cons_text}\n\nagent 的话：\n<<<\n{words}\n>>>")
    claims_purchased = bool(raw.get("claims_purchased"))
    by_check = {r["id"]: r for r in check_results}
    false_verification, unsupported = [], []
    for cc in raw.get("constraint_claims") or []:
        if not isinstance(cc, dict) or cc.get("claim") != "satisfied":
            continue
        quote = str(cc.get("quote") or "")
        if not squash(quote) or squash(quote) not in squash(words):
            continue
        cid = str(cc.get("id"))
        verdict = (by_check.get(cid) or {}).get("verdict")
        status = statuses.get(cid)
        if verdict == "violated" or status in ("unverified", "blind_violation"):
            false_verification.append(cid)
        elif verdict == "unknown":
            unsupported.append(cid)

    price_mismatch = False
    if traj.purchased:
        actual = traj.purchase.get("price")
        try:
            claimed_price = float(raw.get("claimed_price")) if raw.get("claimed_price") is not None else None
        except (TypeError, ValueError):
            claimed_price = None
        if claimed_price is not None and actual is not None:
            price_mismatch = abs(claimed_price - float(actual)) > max(1.0, 0.01 * float(actual))
    return {
        "claims_purchased": claims_purchased,
        "false_purchase_claim": claims_purchased and not traj.purchased,
        "price_mismatch": price_mismatch,
        "false_verification": false_verification,
        "unsupported_claims": unsupported,
        "words": words,
    }


BUTTONS = {"view_description": "description", "view_features": "features", "view_reviews": "reviews", "next_page": "next >",
           "prev_page": "< prev", "back_to_search": "back to search", "buy_now": "buy now"}


def is_invalid(traj: Trajectory, i: int) -> bool:
    """第 i 个环境步骤的点击目标不在执行前页面的可点击项里：环境原地不动，但照样计步。"""
    s = traj.steps[i]
    tool, args = s.get("tool"), s.get("args") or {}
    if tool == "search_products":
        return False
    target = BUTTONS.get(tool) or args.get("asin") or args.get("value") or ""
    clickables = {norm(a) for a in traj.state_before(i).get("actions") or []}
    return norm(target) not in clickables


def first_error(traj: Trajectory, gold: dict, false_purchase_claim: bool) -> dict:
    """失败轨迹第一次偏离成功路径的位置和类型（决定性错误）。步骤号是环境步骤编号（从 1 开始）。
    agent 想对 gold 做正确的事（打开 gold、选 gold 规格），但动作在当前页面无效、被环境静默忽略，优先归为 invalid_action。"""
    gold_asin = str(gold["asin"])
    retrieved = opened = correct = None
    last_gold_page = None
    for i, s in enumerate(traj.steps):
        st = (s.get("result") or {}).get("observation_state") or {}
        page = st.get("page_type")
        if page == "search_results" and retrieved is None and any(str(p.get("asin")) == gold_asin for p in st.get("products") or []):
            retrieved = i
        if page in SHOP_PAGES and str((st.get("product") or {}).get("asin")) == gold_asin:
            opened = i if opened is None else opened
            last_gold_page = i
            if correct is None and options_match(st.get("selected_options"), gold["options"]):
                correct = i

    def step_no(i):
        return traj.steps[i].get("step") if i is not None and 0 <= i < len(traj.steps) else None

    invalid_open_gold = [i for i, s in enumerate(traj.steps)
                         if s.get("tool") == "open_product" and str((s.get("args") or {}).get("asin")) == gold_asin and is_invalid(traj, i)]
    invalid_select_gold = [i for i, s in enumerate(traj.steps)
                           if s.get("tool") == "select_option" and is_invalid(traj, i)
                           and any(gold_value_matches(v, (s.get("args") or {}).get("value", "")) for v in gold["options"])]

    b = traj.buy_index()
    bought = str(traj.purchase.get("asin")) if traj.purchased else None
    if retrieved is None:
        return {"type": "search_miss", "step": None, "searches": sum(1 for s in traj.steps if s.get("tool") == "search_products")}
    if opened is None:
        if invalid_open_gold:
            return {"type": "invalid_action", "intended": "open_gold", "step": step_no(invalid_open_gold[0])}
        return {"type": "missed_candidate", "step": step_no(retrieved)}
    if bought == gold_asin:
        selected = traj.purchase.get("options") or {}
        state = traj.state_before(b)
        axes = {axis: vals for axis, vals in (state.get("available_options") or {}).items()}
        for value in gold["options"]:
            axis = next((a for a, vals in axes.items() if any(gold_value_matches(value, v) for v in vals)), None)
            chosen = selected.get(axis) if axis else None
            if chosen is None:
                if invalid_select_gold:
                    return {"type": "invalid_action", "intended": "select_gold_option", "step": step_no(invalid_select_gold[0]), "option": value}
                return {"type": "missing_option", "step": step_no(b), "option": value}
            if not gold_value_matches(value, chosen):
                last_select = max((i for i in range(b) if traj.steps[i].get("tool") == "select_option"
                                   and norm((traj.steps[i].get("args") or {}).get("value")) == norm(chosen)), default=b)
                return {"type": "wrong_option", "step": step_no(last_select), "option": value, "chosen": chosen}
        return {"type": "other", "step": step_no(b)}
    if correct is None and invalid_select_gold:
        return {"type": "invalid_action", "intended": "select_gold_option", "step": step_no(invalid_select_gold[0])}
    if bought is not None:
        leave = last_gold_page + 1 if last_gold_page is not None and last_gold_page + 1 < len(traj.steps) else last_gold_page
        return {"type": "wrong_rejection", "step": step_no(leave), "options_were_correct": correct is not None}
    if false_purchase_claim:
        attempts = [i for i, s in enumerate(traj.steps) if s.get("tool") == "buy_now"]
        return {"type": "false_completion", "step": step_no(attempts[-1]) if attempts else None, "options_were_correct": correct is not None}
    if (traj.terminal or {}).get("termination_reason") == "max_steps":
        return {"type": "budget_exhausted", "step": step_no(len(traj.steps) - 1), "options_were_correct": correct is not None}
    return {"type": "early_stop", "step": step_no(len(traj.steps) - 1), "options_were_correct": correct is not None}


def interventions(traj: Trajectory) -> dict:
    """插件每次拦截后，模型下一次调用是否原样重复被拦下的动作；收尾提醒之后是否完成了购买。"""
    calls = [(a["turn"], c["name"], json.dumps(c.get("args"), ensure_ascii=False, sort_keys=True)) for a in traj.assistant for c in a["calls"]]
    stats = {}
    for e in traj.harness:
        kind = e.get("type")
        if kind in ("serial_block", "env_error"):
            continue
        entry = stats.setdefault(kind, {"count": 0, "repeated": 0})
        entry["count"] += 1
        if kind in ("guard_block", "audit_block"):
            blocked = (e.get("tool"), json.dumps(e.get("args"), ensure_ascii=False, sort_keys=True))
            nxt = next(((n, a) for t, n, a in calls if t > (e.get("turn") or 0)), None)
            if nxt == blocked and kind == "guard_block":
                entry["repeated"] += 1
    if "finish_hint" in stats:
        stats["finish_hint"]["bought_after"] = traj.purchased
    return stats


BUY_BLOCKS = {"state_verify_block": "下单前对表核对", "confirm_block": "购买前确认"}


def buy_blocks(traj: Trajectory, gold: dict) -> list:
    """harness 拦下的每次购买：被拦时选的是不是正确商品和全对的规格，也就是这次购买放行的话会不会严格成功。"""
    return [{"turn": e.get("turn"), "type": e.get("type"),
             "was_correct": str(e.get("asin")) == str(gold["asin"]) and options_match(e.get("selected"), gold["options"])}
            for e in traj.harness if e.get("type") in BUY_BLOCKS]

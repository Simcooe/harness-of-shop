#!/usr/bin/env python3
"""第二步：拿买到的商品数据逐条核对需求约束。

证据卡只来自商品库和环境的购买记录，不使用 agent 写的任何文字。预算约束用规则判定，其余交给 LLM，
且 LLM 的每个结论都必须引用证据卡中真实存在的原文，否则改判为“无法确认”。

用法（gold 自检）：python3 eval/checker.py --gold-selfcheck [--split dev]
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "third_party/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/items_eval_train.json"
PRODUCTS_CACHE = ROOT / "data" / "cache" / "products_min.json"
GOLD_PATH = ROOT / "data" / "private" / "gold.jsonl"
RUBRIC_DIR = ROOT / "data" / "rubrics"
MODEL = "deepseek-v4-flash"
APPROX_TOLERANCE = 0.20
FIELDS = {
    "title": "标题",
    "shop": "店铺",
    "category": "类目",
    "attributes": "属性",
    "description": "商品描述",
    "price_range": "价格区间",
    "available_options": "可选规格",
    "selected_options": "已选规格",
    "price": "成交价",
}

SYSTEM = """你是电商商品核验员。根据“商品证据卡”判断这件已购买的商品是否满足每一条需求约束，输出 JSON。

规则：
1. 只依据证据卡。可以做显而易见的语义判断（如“杀虫剂”属于“杀虫药”，“2瓶”就是“两瓶”，“灰色”不是“黑色”），但不能凭常识补充证据卡里没写的具体事实。
2. 品类（category）按商品实际是什么来判断，类目名称只作参考：商品属于需求所说的那类东西，或是它的子类、近义叫法，就算满足。
3. 需求里带“左右、约、大概、差不多”的数值（尺寸、容量、重量、数量等），在 numeric 里给出需求数值 required 和商品数值 actual（换算成同一单位的纯数字），并设 approx 为 true，程序会按 20% 的容差判定；“20多厘米”“3升多”这类写法表示一个区间（20 到 30、3 到 4），required 填区间中点。近义颜色（如墨色和黑色、奶白和米白）算满足，明显不同的颜色才算违反。规格值没有写明数量时，按 1 件（1 瓶、1 个）理解。
4. verdict 取值：
   - satisfied：证据卡中有信息表明满足
   - violated：证据卡中有信息与约束明显矛盾
   - unknown：证据卡没有足够信息判断（没提到就是 unknown，不是 violated）
5. spec 类约束看“已选规格”。如果已选规格不满足，再检查“可选规格”里是否有满足的值，有就把那个值原样填进 available_value。
6. satisfied 和 violated 必须给出 evidence_field 和 evidence_quote：evidence_field 是证据卡字段名（title、shop、category、attributes、description、price_range、available_options、selected_options、price 之一），evidence_quote 是该字段中原样摘录的连续片段。
7. reason 用一句话说明。

输出格式：{"results": [{"id": "c1", "verdict": "satisfied|violated|unknown", "evidence_field": "...", "evidence_quote": "...", "available_value": "", "numeric": {"required": null, "actual": null, "approx": false}, "reason": "..."}]}"""


def squash(text) -> str:
    return re.sub(r"[\s，。,.！!？?、；;：:“”\"'（）()【】\[\]|/]", "", str(text)).lower()


def load_products() -> dict:
    if PRODUCTS_CACHE.exists():
        return json.loads(PRODUCTS_CACHE.read_text(encoding="utf-8"))
    items = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    products = {}
    for it in items:
        pricing = it.get("pricing") or []
        options = {
            axis: [{"value": str(v.get("value", "")).strip(), "price": v.get("price")} for v in (values or [])]
            for axis, values in (it.get("customization_options") or {}).items()
        }
        products[str(it["asin"])] = {
            "title": it.get("title") or "",
            "shop": it.get("shop_name") or "",
            "category": it.get("category") or "",
            "attributes": [str(a) for a in (it.get("attribute") or [])],
            "description": it.get("full_description") or "",
            "price_range": " to ".join(str(p) for p in pricing[:2]) if pricing else "",
            "options": options,
        }
    PRODUCTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTS_CACHE.write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
    return products


def build_card(product: dict, selected: dict, price) -> dict:
    available = {axis: [o["value"] for o in opts] for axis, opts in product["options"].items()}
    return {
        "title": product["title"],
        "shop": product["shop"],
        "category": product["category"],
        "attributes": "；".join(product["attributes"]),
        "description": product["description"],
        "price_range": product["price_range"],
        "available_options": json.dumps(available, ensure_ascii=False),
        "selected_options": json.dumps(selected or {}, ensure_ascii=False),
        "price": "" if price is None else str(price),
    }


def check_budget(constraint: dict, price):
    spec = constraint.get("budget")
    if not spec or price is None:
        return {"id": constraint["id"], "verdict": "unknown", "method": "rule", "reason": "无法解析预算或没有成交价"}
    p = float(price)
    if spec["op"] == "upper":
        ok, rule = p <= spec["value"], f"成交价 {p:g} ≤ {spec['value']:g}"
    elif spec["op"] == "approx":
        limit = spec["value"] * (1 + APPROX_TOLERANCE)
        ok, rule = p <= limit, f"成交价 {p:g} ≤ {spec['value']:g}×{1 + APPROX_TOLERANCE:g}"
    elif spec["op"] == "lower":
        ok, rule = p >= spec["value"], f"成交价 {p:g} ≥ {spec['value']:g}"
    else:
        ok = spec["low"] * (1 - APPROX_TOLERANCE) <= p <= spec["high"] * (1 + APPROX_TOLERANCE)
        rule = f"成交价 {p:g} 在 [{spec['low']:g}, {spec['high']:g}] 放宽 20% 的区间内"
    return {
        "id": constraint["id"],
        "verdict": "satisfied" if ok else "violated",
        "method": "rule",
        "evidence_field": "price",
        "evidence_quote": f"{p:g}",
        "reason": rule if ok else f"不满足：{rule}",
    }


def check_purchase(rubric: dict, product: dict, selected: dict, price) -> list:
    card = build_card(product, selected, price)
    results = {}
    llm_constraints = []
    for c in rubric["constraints"]:
        if c["type"] == "budget":
            results[c["id"]] = check_budget(c, price)
        else:
            llm_constraints.append(c)

    if llm_constraints:
        card_text = "\n".join(f"{FIELDS[k]}（{k}）：{v}" for k, v in card.items())
        cons_text = "\n".join(f"{c['id']} [{c['type']}] {c['description']}" for c in llm_constraints)
        raw = chat_json(MODEL, SYSTEM, f"商品证据卡：\n{card_text}\n\n需求约束：\n{cons_text}")
        by_id = {str(r.get("id")): r for r in raw.get("results", []) if isinstance(r, dict)}
        for c in llm_constraints:
            r = by_id.get(c["id"], {})
            verdict = r.get("verdict") if r.get("verdict") in ("satisfied", "violated", "unknown") else "unknown"
            field, quote = str(r.get("evidence_field") or ""), str(r.get("evidence_quote") or "")
            item = {"id": c["id"], "verdict": verdict, "method": "llm", "evidence_field": field, "evidence_quote": quote,
                    "available_value": str(r.get("available_value") or ""), "reason": str(r.get("reason") or "")}
            numeric = r.get("numeric") if isinstance(r.get("numeric"), dict) else {}
            try:
                required, actual = float(numeric.get("required")), float(numeric.get("actual"))
            except (TypeError, ValueError):
                required = actual = None
            if numeric.get("approx") and required and actual is not None and verdict != "unknown":
                ok = abs(actual - required) / required <= APPROX_TOLERANCE
                item["verdict"], item["method"] = ("satisfied" if ok else "violated"), "llm+numeric"
                item["reason"] = f"需求约 {required:g}，商品 {actual:g}，偏差 {abs(actual - required) / required:.0%}"
            if item["verdict"] != "unknown" and (field not in card or not squash(quote) or squash(quote) not in squash(card[field])):
                item["verdict"], item["quote_invalid"] = "unknown", True
            results[c["id"]] = item
    return [results[c["id"]] for c in rubric["constraints"]]


VERIFIABLE_TYPES = {"category", "brand", "model", "spec", "budget"}


def summarize_check(rubric: dict, results: list, skip_ids=()) -> dict:
    """三档可接受口径（服务类约束都不计入）：
    strict：所有 hard 约束都满足；
    core：品类、品牌、型号、规格、预算这些商品数据能核实的 hard 约束必须满足，功能类 hard 约束不被判违反即可；
    lenient：没有任何 hard 约束被判违反。"""
    by_id = {r["id"]: r for r in results}
    hard = [c for c in rubric["constraints"] if c["hardness"] == "hard" and c["type"] != "service" and c["id"] not in skip_ids]
    verdict = {c["id"]: by_id[c["id"]]["verdict"] for c in hard}
    return {
        "acceptable": all(v == "satisfied" for v in verdict.values()),
        "acceptable_core": all(
            verdict[c["id"]] == "satisfied" if c["type"] in VERIFIABLE_TYPES else verdict[c["id"]] != "violated" for c in hard
        ),
        "acceptable_lenient": all(v != "violated" for v in verdict.values()),
        "hard_violated": [c["id"] for c in hard if verdict[c["id"]] == "violated"],
        "hard_unknown": [c["id"] for c in hard if verdict[c["id"]] == "unknown"],
        "violated_types": sorted({c["type"] for c in hard if verdict[c["id"]] == "violated"}),
    }


def variant_price(product: dict, selected: dict):
    """与环境 variant-price-v1 的规则一致：只有一个规格项的取值价格不同时，按该项所选值定价；
    多个规格项都影响价格时视为无法确定。"""
    effective, all_prices = [], set()
    for axis, opts in product["options"].items():
        prices = {float(o["price"]) for o in opts if o.get("price") is not None}
        all_prices |= prices
        if len(prices) > 1:
            effective.append(axis)
    if len(effective) > 1:
        return None
    if len(effective) == 1:
        axis = effective[0]
        value = selected.get(axis)
        hit = next((o for o in product["options"][axis] if value is not None and squash(o["value"]) == squash(value)), None)
        return float(hit["price"]) if hit and hit.get("price") is not None else None
    if len(all_prices) == 1:
        return next(iter(all_prices))
    nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", product["price_range"])}
    return next(iter(nums)) if not product["options"] and len(nums) == 1 else None


def gold_selection(product: dict, gold_options: list):
    """把 gold 规格值映射回各规格项，并按环境规则算出该规格组合的价格。"""
    selected = {}
    for value in gold_options:
        for axis, opts in product["options"].items():
            hit = next((o for o in opts if squash(o["value"]) == squash(value)), None)
            if hit:
                selected[axis] = hit["value"]
                break
    return selected, variant_price(product, selected)


def load_rubric(task_id: int):
    path = RUBRIC_DIR / f"{task_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-selfcheck", action="store_true")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.gold_selfcheck:
        parser.error("目前只支持 --gold-selfcheck")

    products = load_products()
    gold = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [g for g in gold if g["split"] == args.split and load_rubric(g["task_id"])]

    def work(g):
        rubric = load_rubric(g["task_id"])
        product = products[str(g["asin"])]
        selected, price = gold_selection(product, g["options"])
        results = check_purchase(rubric, product, selected, price)
        # gold 的规格组合有时按环境规则也无法定价，此时预算约束不参与自检
        skip = {c["id"] for c in rubric["constraints"] if c["type"] == "budget"} if price is None else set()
        return {"task_id": g["task_id"], "selected": selected, "price": price, "results": results,
                **summarize_check(rubric, results, skip)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(work, gold))
    out = ROOT / "data" / "cache" / f"gold_selfcheck_{args.split}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    n = len(rows)
    print(f"gold 自检（{args.split}，{n} 题）")
    print(f"  strict  （hard 全部满足）：{sum(r['acceptable'] for r in rows)}/{n}")
    print(f"  core    （可核实的 hard 满足，功能类不违反）：{sum(r['acceptable_core'] for r in rows)}/{n}")
    print(f"  lenient （没有 hard 被判违反）：{sum(r['acceptable_lenient'] for r in rows)}/{n}")
    print(f"  有 hard 约束被判违反的题：{[r['task_id'] for r in rows if r['hard_violated']]}")
    print(f"  结果写入 {out}")


if __name__ == "__main__":
    main()

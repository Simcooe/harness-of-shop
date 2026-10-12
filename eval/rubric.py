#!/usr/bin/env python3
"""第一步：从 query 抽取需求约束（rubric），只看 query、不看 gold，结果冻结到 data/rubrics/。

用法：python3 eval/rubric.py [--split dev|test|all]
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
GOLD_PATH = ROOT / "data" / "private" / "gold.jsonl"
RUBRIC_DIR = ROOT / "data" / "rubrics"
MODEL = "deepseek-v4-flash"
TYPES = {"category", "brand", "model", "function", "spec", "budget", "service"}

SYSTEM = """你是电商需求分析员。把用户的一条购物需求拆成可以逐条核验的原子约束，输出 JSON。

规则：
1. 只依据需求原文。不要补充用户没说的要求，不要猜测具体商品，不要把常识当成要求。
2. 每条约束只包含一个要点。type 取值：
   - category：要买的是什么商品（品类），每条需求都有且只有一条
   - brand：品牌
   - model：型号、系列
   - function：商品本身的特征，如功能、用途、适用人群或场景、材质、风格、工艺、成分、参数
   - spec：需要在商品规格里选择的项，如颜色、尺码、容量、重量、数量或几件装、套餐、口味、款式
   - budget：价格或预算
   - service：发货、物流、售后、正品保证、库存、店铺资质等服务类要求
3. hardness：用户明确提出的要求为 hard；用“最好、希望、尽量、可以的话、优先、如果有”等表达的偏好为 soft。预算一律为 hard。
4. query_quote：从需求原文中原样摘录支持这条约束的最短片段，必须是原文的连续子串，不能改写。
5. description：用一句简短的中文写清这条约束。
6. 预算约束额外给出 budget_normalized：把口语写法换成只用阿拉伯数字的标准写法，并保留限定词。例如“三张票子差不多”写成“约300元”，“不超过两千”写成“2000元以内”，“一百左右”写成“约100元”，“不低于3元”写成“不低于3元”，“三五百”写成“300到500元”。

输出格式：{"constraints": [{"id": "c1", "type": "...", "hardness": "hard|soft", "description": "...", "query_quote": "...", "budget_normalized": "仅预算约束需要"}]}"""

CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def cn_to_number(text: str):
    """把“六百”“一千二”“两万”“3千”“1.5万”这类写法转成数字，转不了返回 None。"""
    text = text.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([十百千万]?)", text)
    if m:
        return float(m.group(1)) * CN_UNITS.get(m.group(2), 1)
    if not text or any(ch not in CN_DIGITS and ch not in CN_UNITS for ch in text):
        return None
    total, section, digit, last_unit = 0, 0, None, 1
    for ch in text:
        if ch in CN_DIGITS:
            digit = CN_DIGITS[ch]
        else:
            unit = CN_UNITS[ch]
            if unit == 10000:
                section = (section + (digit or 0)) * unit
                total += section
                section, digit = 0, None
            else:
                section += (digit if digit is not None else 1) * unit
                digit = None
            last_unit = unit
    if digit is not None:
        # “一千二”表示 1200：末尾裸数字按上一个单位的十分之一计
        section += digit * (last_unit // 10 if last_unit >= 100 else 1)
    return float(total + section)


NUM = r"(\d+(?:\.\d+)?\s*[十百千万]?|[零〇一二两三四五六七八九十百千万]+)"


def parse_budget(quote: str):
    """从预算原文解析出比较方式：upper（硬上限）、approx（约数，上浮 20% 以内）、lower（下限）、range（区间）。"""
    q = quote.replace(" ", "")
    m = re.search(NUM + r"\s*(?:元|块|rmb)?\s*[-~到至]\s*" + NUM, q, re.I)
    if m:
        lo, hi = cn_to_number(m.group(1)), cn_to_number(m.group(2))
        if lo is not None and hi is not None:
            return {"op": "range", "low": lo, "high": hi}
    m = re.search(r"(?:不低于|不少于|至少|起码)\s*" + NUM, q) or re.search(NUM + r"\s*(?:元|块)?\s*(?:以上|起)", q)
    if m and cn_to_number(m.group(1)) is not None:
        return {"op": "lower", "value": cn_to_number(m.group(1))}
    m = re.search(r"(?:不超过|不要超过|不高于|最多|上限|控制在)\s*" + NUM, q) or re.search(
        NUM + r"\s*(?:元|块)?\s*(?:以内|以下|之内|内)", q
    )
    if m and cn_to_number(m.group(1)) is not None and not re.search(r"左右|上下|出头|大概|大约|差不多", q):
        return {"op": "upper", "value": cn_to_number(m.group(1))}
    m = re.search(NUM + r"\s*(?:元|块|rmb)?", q, re.I)
    if m and cn_to_number(m.group(1)) is not None:
        return {"op": "approx", "value": cn_to_number(m.group(1))}
    return None


def squash(text: str) -> str:
    return re.sub(r"[\s，。,.！!？?、；;：:“”\"'（）()]", "", str(text))


def extract(task: dict) -> dict:
    query = task["instruction"]
    raw = chat_json(MODEL, SYSTEM, f"购物需求：{query}")
    constraints, dropped = [], []
    for i, c in enumerate(raw.get("constraints", []), start=1):
        quote = str(c.get("query_quote", ""))
        ctype = str(c.get("type", "")).strip()
        if ctype not in TYPES or not quote or squash(quote) not in squash(query):
            dropped.append(c)
            continue
        item = {
            "id": f"c{i}",
            "type": ctype,
            "hardness": "hard" if ctype == "budget" else ("soft" if c.get("hardness") == "soft" else "hard"),
            "description": str(c.get("description", "")).strip(),
            "query_quote": quote,
        }
        if ctype == "budget":
            normalized = str(c.get("budget_normalized") or "").strip()
            item["budget_normalized"] = normalized
            item["budget"] = parse_budget(normalized) or parse_budget(quote)
        constraints.append(item)
    return {"task_id": task["task_id"], "query": query, "model": MODEL, "constraints": constraints, "dropped": dropped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["dev", "test", "all"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.split != "all":
        tasks = [t for t in tasks if t["split"] == args.split]
    RUBRIC_DIR.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (RUBRIC_DIR / f"{t['task_id']}.json").exists()]

    def work(task):
        rubric = extract(task)
        (RUBRIC_DIR / f"{task['task_id']}.json").write_text(json.dumps(rubric, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rubric

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(work, todo))
    n_dropped = sum(len(r["dropped"]) for r in results)
    n_cons = sum(len(r["constraints"]) for r in results)
    print(f"新抽取 {len(results)} 题（已存在 {len(tasks) - len(todo)} 题），约束 {n_cons} 条，因原文校验不通过丢弃 {n_dropped} 条")


if __name__ == "__main__":
    main()

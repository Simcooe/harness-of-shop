"""失败原因诊断：规则（process.first_error）只能指出第一次偏离成功路径的是哪一步，这里回答为什么偏离。

原因按几条能核对的事实由规则判定，不让模型直接挑原因：模型一次拍板时，"没问清需求"和"看到过正确商品没回头"
之间经常判反（要同时理清用户每回合说了什么、两件商品逐条比较、再归类，而且它看得到标准答案）。用到的事实：
- 每条需求，买的那件和正确商品各自满不满足：评测器的核对结果，和标准答案自检（data/cache/gold_selfcheck_*.jsonl）；
- 每条需求用户第几回合第一次说出口：小模型对照用户原话判断，必须引用原话，原话在那段对话里找不到的不算；
- 正确商品第几回合出现在搜索结果里、有没有被打开，harness 拦过哪些购买：日志；
- 买对了商品但规格不对时，按用户说过的话和预算能不能确定该选哪个：评审模型只回答选哪个；
- 正确商品这次没搜到时，单轮实验（agent 一开始就知道完整需求）的运行有没有搜到过它，用的搜索词按用户当时说过的话想不想得到
  （评审模型逐词判断）。
理由按事实套模板写。"追着查页面上没有的信息""回合用完被迫下单"是耗掉回合的方式，作为附加标签，不和根本原因抢。
"""
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import JUDGE_MODEL  # noqa: E402
from llm import chat_json  # noqa: E402
from process import BUY_BLOCKS, buy_blocks  # noqa: E402
from traj import gold_value_matches, load_jsonl, norm, squash  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
SAID_MODEL = "deepseek-v4-flash"
MIN_REFERENCE_RUNS = 3
# 搜索能匹配到的字段；价格和规格要打开商品才看得到
SEARCHABLE = {"title", "attributes", "category", "shop"}
# 搜索结果每一行显示的字段（标题、价格、属性），其余要打开商品才看得到
IN_RESULTS = {"title", "attributes", "price"}
HARD_CLOSE = {"closing_hard", "state_close_hard"}
ACT_TURNS = 3  # 用户说出条件后，搜索、打开、下单至少要这么多回合；离"必须下单"不到这么多回合才说的，算说得太晚
SUBPAGE_CHASE = 6  # 看一件商品的三个详情子页是 3 次，看两件以上还在看才算追查

CAUSES = {
    "没问清需求": f"正确商品满足、买的那件不满足的条件，用户到最后都没说过（或者说的时候离必须下单已经不到 {ACT_TURNS} 个回合）；"
              "买对了商品但该选哪个规格，按用户说过的话和预算确定不了",
    "看到过正确商品没回头": "用户说过某条条件，正确商品满足、买的那件不满足；用户说过之后，正确商品出现在搜索结果里或被打开过，agent 没有选它",
    "搜索词不对": "用户说过某条条件，正确商品满足、买的那件不满足，但之后正确商品一直没出现在搜索结果里，而按用户说过的话想得到能搜出它的词",
    "规格没点": "买的就是正确商品，有规格项没点（只有一个选项，或者按用户说的能确定选哪个）",
    "规格选错": "买的就是正确商品，规格项点了别的选项，而按用户说过的话或预算能确定该选哪个",
    "替代品也满足需求": "按每一条需求核对，买的那件都不比正确商品差（正确商品满足的它也满足），只是不是标注的那一件",
    "正确商品难搜到": "正确商品这次没搜到，知道完整需求的 agent（单轮实验多次运行）也从没搜到过",
    "系统拦下了正确的购买": "agent 已经选好正确商品和全对的规格去下单，被 harness 的检查拦下，之后没能买成",
    "动作无效": "想打开正确商品或选正确规格，但动作在当前页面无效，被环境忽略",
    "标准答案有问题": "正确商品不满足用户说过的某条要求（比如超预算），买的那件满足；按用户说的，不选正确商品没错",
    "没买成": "到最后都没有下单（环境步数或回合用完，或者 agent 自己停了）",
    "其他": "缺少判断所需的数据",
}
TAGS = {
    "回合用完被迫下单": "系统提示过“现在必须下单”",
    "追查页面上没有的信息": f"看了 {SUBPAGE_CHASE} 次以上没有内容的详情子页",
    "标注可能有问题": "标准答案自检没通过",
}

SAID_SYSTEM = """给你一条网购需求拆出的条件清单，和用户在对话里按回合说过的话（回合 0 是开场白）。
对清单里的每一条，找出用户第一次说出这条条件的具体要求是在哪一回合：
- 意思对上就算，措辞可以不同；用提问的方式说出的要求也算（比如“这个是原装的吗”说明用户要原装）。
- 只是被问到、回答“都行”“没要求”，或者只说了大方向、没说出这条的具体要求（比如条件是“颜色为墨色”，用户只说“颜色好看就行”），不算。
- 用户没说过的条件不要输出。
每条附上用户那一回合原话里的一小段作为依据，必须是原文里连续的一段，不要改写。
输出 JSON：{"said": [{"id": "c1", "turn": 2, "quote": "预算大概在90元左右"}]}"""

SPEC_SYSTEM = """给你用户在购物对话里按回合说过的话、商品标题，和这件商品某个规格项的全部选项（附各选项价格）。
只根据用户说过的话（包括预算）判断用户要的是哪个选项：
- 能确定是哪一个，就填那个选项的原文；用户点名要的选项（比如说出了颜色名、型号）就是那一个；
- 选项名里没写到的功能，如果商品标题里写了，算每个选项都有；
- 用户没说过相关的要求，或者几个选项都说得通，就填空字符串。
不要因为某个选项看起来更常见就选它。
输出 JSON：{"choice": "选项原文或空字符串", "basis": "一句话依据：用户原话或预算"}"""

QUERY_SYSTEM = """给你用户在购物对话里按回合说过的话，和几条搜索词。对每条搜索词，逐个看里面的词：
- 用户说过的意思（换个说法也算），或者商品的品类名、通用叫法：不算新信息；
- 用户没说过的条件（比如颜色、款式、适用人群、规格、促销方式、品牌）：列进这条搜索词的 unsaid。
输出 JSON：{"queries": [{"query": "原搜索词", "unsaid": ["男女同款"]}]}"""


@lru_cache(maxsize=None)
def _gold_checks() -> dict:
    """标准答案自检：每道题里正确商品对每条需求的核对结果。"""
    checks = {}
    for path in sorted((ROOT / "data" / "cache").glob("gold_selfcheck_*.jsonl")):
        for row in load_jsonl(path):
            checks[row["task_id"]] = {r["id"]: r for r in row["results"]}
    return checks


@lru_cache(maxsize=None)
def _reference_runs() -> tuple:
    """单轮实验的运行：run_config 里既没有多轮协议也没有多轮需求文件，agent 一开始就拿到完整需求。"""
    runs = []
    for path in sorted(RUNS.glob("*/run_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if not config.get("protocol") and not config.get("multiturn"):
            runs.append(path.parent)
    return tuple(runs)


def full_info_reference(task_id: int, gold_asin: str) -> dict:
    """这道题在单轮实验的各次运行里，有几次搜到过正确商品，搜到时用的搜索词。"""
    runs, queries = 0, []
    for run in _reference_runs():
        steps = load_jsonl(run / f"task_{task_id}" / "env_steps.jsonl")
        if not steps:
            continue
        runs += 1
        for s in steps:
            st = (s.get("result") or {}).get("observation_state") or {}
            if st.get("page_type") == "search_results" and gold_asin in {str(p.get("asin")) for p in st.get("products") or []}:
                queries.append(str(st.get("query")))
                break
    return {"runs": runs, "retrieved": len(queries), "queries": [q for q, _ in Counter(queries).most_common(3)]}


def statements(traj) -> list:
    """用户说过的话：(回合, 原文)，开场白算第 0 回合。"""
    return [(0, traj.agent_query)] + [(int(q.get("turn") or 0), str(q.get("answer") or "")) for q in traj.asks]


def user_said(traj, rubric: dict) -> dict:
    """每条需求用户第一次说出口的回合和原话；没说过的不在结果里。"""
    cons = rubric["constraints"]
    if traj.agent_query == traj.query:  # 单轮：需求一开始就完整给出
        return {c["id"]: {"turn": 0, "quote": c.get("query_quote") or ""} for c in cons}
    stmts = statements(traj)
    user = ("条件清单：\n" + "\n".join(f"{c['id']} {c['description']}" for c in cons)
            + "\n\n用户说过的话：\n" + "\n".join(f"回合 {t}：{text}" for t, text in stmts))
    raw = chat_json(SAID_MODEL, SAID_SYSTEM, user, max_tokens=2000)
    ids, said = {c["id"] for c in cons}, {}
    for item in raw.get("said") or []:
        cid, quote = item.get("id"), str(item.get("quote") or "")
        turns = [t for t, text in stmts if squash(quote) and squash(quote) in squash(text)]
        if cid not in ids or not turns:
            continue
        try:
            claimed = int(item.get("turn"))
        except (TypeError, ValueError):
            claimed = None
        turn = claimed if claimed in turns else min(turns)
        if cid not in said or turn < said[cid]["turn"]:
            said[cid] = {"turn": turn, "quote": quote}
    return said


def gold_sightings(traj, gold_asin: str) -> list:
    """正确商品每次出现在 agent 面前：在某页搜索结果里，或者详情页被打开。"""
    seen = []
    for s in traj.steps:
        st = (s.get("result") or {}).get("observation_state") or {}
        page = st.get("page_type")
        if page == "search_results":
            asins = [str(p.get("asin")) for p in st.get("products") or []]
            if gold_asin in asins:
                seen.append({"step": s.get("step"), "turn": s.get("turn") or 0, "where": f"出现在搜索结果第 {asins.index(gold_asin) + 1} 个"})
        elif page in ("product_detail", "information_subpage") and str((st.get("product") or {}).get("asin")) == gold_asin:
            seen.append({"step": s.get("step"), "turn": s.get("turn") or 0, "where": "被打开"})  # 详情页或它的子页
    return seen


def spec_problems(traj, gold: dict, products: dict) -> list:
    """买对了商品时，哪些规格项没点、哪些点错了。"""
    options = (products.get(str(gold["asin"])) or {}).get("options") or {}
    selected = traj.purchase.get("options") or {}
    problems = []
    for gv in gold.get("options") or []:
        axis = next((a for a, vals in options.items() if any(gold_value_matches(gv, v["value"]) for v in vals)), None)
        if axis is None:
            continue
        sel = next((v for k, v in selected.items() if norm(k) == norm(axis)), None)
        if sel is not None and gold_value_matches(gv, sel):
            continue
        problems.append({"axis": axis, "gold": gv, "selected": sel, "values": options[axis]})
    return problems


def _match_option(choice: str, values: list) -> str:
    """模型答的选项要对回选项原文：完全一致优先，否则取唯一一个互相包含的（比如答“L号”，选项是“L”）。"""
    if not choice:
        return ""
    exact = [v["value"] for v in values if norm(v["value"]) == norm(choice)]
    if exact:
        return exact[0]
    near = [v["value"] for v in values if norm(v["value"]) in norm(choice) or norm(choice) in norm(v["value"])]
    return near[0] if len(near) == 1 else choice


def spec_choice(traj, title: str, axis: str, values: list) -> dict:
    """只按用户在下单前说过的话和预算，这个规格项该选哪个。"""
    buy = traj.buy_turn() or 10**9
    stmts = [(t, text) for t, text in statements(traj) if t < buy]
    user = ("用户说过的话：\n" + "\n".join(f"回合 {t}：{text}" for t, text in stmts)
            + f"\n\n商品标题：{title}\n规格项「{axis}」的选项：\n" + "\n".join(f"- {v['value']}（{v.get('price')} 元）" for v in values))
    raw = chat_json(JUDGE_MODEL, SPEC_SYSTEM, user, max_tokens=800, timeout=600)
    return {"choice": _match_option(str(raw.get("choice") or ""), values), "basis": str(raw.get("basis") or "")}


def query_check(traj, queries: list, cutoff: int) -> dict:
    """知道完整需求的 agent 搜到正确商品用的词，按用户在 cutoff 回合之前说过的话能不能想到。"""
    stmts = [(t, text) for t, text in statements(traj) if t <= cutoff]
    user = ("用户说过的话：\n" + "\n".join(f"回合 {t}：{text}" for t, text in stmts)
            + "\n\n搜索词：\n" + "\n".join(f"- {q}" for q in queries))
    raw = chat_json(JUDGE_MODEL, QUERY_SYSTEM, user, max_tokens=1000, timeout=600)
    rows = [r for r in raw.get("queries") or [] if isinstance(r, dict)]
    formable = next((str(r.get("query")) for r in rows if not r.get("unsaid")), "")
    unsaid = list(dict.fromkeys(str(w) for r in rows for w in r.get("unsaid") or []))
    return {"formable": formable, "unsaid": unsaid}


def _tags(traj, rec: dict) -> list:
    tags = []
    if any(e.get("type") in HARD_CLOSE for e in traj.harness):
        tags.append("回合用完被迫下单")
    if (rec.get("label_counts") or {}).get("view_subpage", 0) >= SUBPAGE_CHASE:
        tags.append("追查页面上没有的信息")
    if rec.get("gold_conflict"):
        tags.append("标注可能有问题")
    return tags


def _spec_cause(traj, gold: dict, products: dict) -> tuple:
    """买对了商品、规格不全对：没点的单选项算规格没点；其余看按用户的话或预算能不能确定该选哪个。"""
    labels, notes, facts = [], [], []
    for p in spec_problems(traj, gold, products):
        single = len(p["values"]) == 1
        pick = None if single else spec_choice(traj, (products.get(str(gold["asin"])) or {}).get("title", ""), p["axis"], p["values"])
        if pick:
            pick["basis"] = pick["basis"].rstrip("。.，, ")
        knowable = single or gold_value_matches(p["gold"], pick["choice"])
        facts.append({**{k: p[k] for k in ("axis", "gold", "selected")}, "single": single, "pick": pick})
        if p["selected"] is None and knowable:
            labels.append("规格没点")
            notes.append(f"规格「{p['axis']}」没点（" + (f"只有「{p['gold']}」一个选项）" if single else f"按{pick['basis']}，应该选「{p['gold']}」）"))
        elif knowable:
            labels.append("规格选错")
            notes.append(f"规格「{p['axis']}」选了「{p['selected']}」，按{pick['basis']}，应该选「{p['gold']}」")
        else:
            labels.append("没问清需求")
            what = "没点" if p["selected"] is None else f"选了「{p['selected']}」"
            why = pick["basis"] if pick and pick["basis"] else "用户没说过，按预算也推不出来"
            notes.append(f"规格「{p['axis']}」{what}，应该是「{p['gold']}」，但按用户说的确定不了该选哪个（{why}），应该先问清")
    if not labels:
        return "其他", [], "买的是正确商品，但对不上规格项", facts
    order = ["没问清需求", "规格选错", "规格没点"]
    cause = min(labels, key=order.index)
    return cause, sorted(set(labels) - {cause}, key=order.index), "买的就是正确商品，但" + "；".join(notes) + "。", facts


def diagnose(traj, rubric: dict, gold: dict, rec: dict, products: dict) -> dict:
    gold_asin = str(gold["asin"])
    out = {"cause": "其他", "secondary": [], "tags": _tags(traj, rec), "steps": [], "explanation": "", "facts": {}}

    stopped = next((b for b in buy_blocks(traj, gold) if b["was_correct"]), None)
    if stopped:
        return {**out, "cause": "系统拦下了正确的购买", "steps": [f"回合 {stopped['turn']}"],
                "explanation": f"第 {stopped['turn']} 回合 agent 已经选好正确商品和规格去下单，被系统的{BUY_BLOCKS[stopped['type']]}拦下，之后没能买成。"}
    fe = rec.get("first_error") or {}
    if fe.get("type") == "invalid_action":
        return {**out, "cause": "动作无效", "steps": [f"S{fe.get('step')}"],
                "explanation": f"S{fe.get('step')} 想打开正确商品或选正确规格，但动作在当时的页面上无效，被环境忽略。"}

    if not traj.purchased:
        why = ("环境步数用完" if len(traj.steps) >= 35 else "回合用完" if len(traj.assistant) >= 40 else "agent 自己停了")
        seen = gold_sightings(traj, gold_asin)
        where = f"正确商品在 S{seen[0]['step']}（第 {seen[0]['turn']} 回合）{seen[0]['where']}过" if seen else "正确商品没出现过"
        return {**out, "cause": "没买成", "explanation": f"到最后都没有下单（{why}）；{where}。"}

    if rec.get("bought_gold"):
        cause, secondary, text, facts = _spec_cause(traj, gold, products)
        return {**out, "cause": cause, "secondary": secondary, "explanation": text, "facts": {"spec": facts}}

    checks = _gold_checks().get(traj.task_id)
    if checks is None:
        return {**out, "explanation": "缺少标准答案的核对结果。"}
    desc = {c["id"]: c["description"] for c in rubric["constraints"]}
    bought = {r["id"]: r.get("verdict") for r in rec.get("check") or []}
    better = [cid for cid in desc if (checks.get(cid) or {}).get("verdict") == "satisfied" and bought.get(cid) != "satisfied"]
    if not better:
        return {**out, "cause": "替代品也满足需求",
                "explanation": "按每一条需求核对，正确商品满足的，买的那件也都满足，只是不是标注的那一件。"}

    seen = gold_sightings(traj, gold_asin)
    ref = None
    if not seen:
        ref = full_info_reference(traj.task_id, gold_asin)
        out["reference"] = ref
        if ref["runs"] >= MIN_REFERENCE_RUNS and not ref["retrieved"]:
            return {**out, "cause": "正确商品难搜到",
                    "explanation": f"正确商品这次没搜到；一开始就知道完整需求的 agent（单轮实验 {ref['runs']} 次运行）也从没搜到过它。"}

    said = user_said(traj, rubric)
    late = next((e.get("turn") for e in traj.harness if e.get("type") in HARD_CLOSE), None)
    cutoff = late - ACT_TURNS if late is not None else 10**9
    in_time = {cid for cid in said if said[cid]["turn"] <= cutoff}
    worse = [cid for cid in desc if cid in in_time and (checks.get(cid) or {}).get("verdict") == "violated" and bought.get(cid) == "satisfied"]
    out["facts"] = {"better": better, "worse": worse, "said": said, "late_turn": late, "sightings": seen[:5]}
    if worse:
        cid = worse[0]
        return {**out, "cause": "标准答案有问题", "steps": [f"回合 {said[cid]['turn']}"],
                "explanation": f"用户第 {said[cid]['turn']} 回合说了「{said[cid]['quote']}」（{desc[cid]}），正确商品不满足这条，买的那件满足；"
                               f"按用户说的，不选正确商品没错。"}
    miss = {"violated": "不满足", "unknown": "确认不了"}

    def bought_word(cid):
        return f"买的那件{miss.get(bought.get(cid), '不满足')}"

    def said_text(cid):
        return f"用户第 {said[cid]['turn']} 回合说了「{said[cid]['quote']}」（{desc[cid]}）"

    def ask_more(cids, note=""):
        never = [desc[c] for c in cids if c not in said]
        parts = ([f"{'、'.join(never)}，用户到最后都没说过"] if never else []) + \
                [f"{desc[c]}，用户第 {said[c]['turn']} 回合才说，离必须下单已经不到 {ACT_TURNS} 个回合" for c in cids if c in said]
        verdict = "不满足" if all(bought.get(c) == "violated" for c in cids) else "不满足或确认不了"
        return {**out, "cause": "没问清需求", "explanation": f"正确商品满足、买的那件{verdict}的条件里：" + "；".join(parts) + note + "。"}

    # 正确商品出现过：用户说过某条正确商品更好的条件之后它又出现，是没回头。买的那件明确不满足的条件
    # 比只是"页面上看不出来"的更能区分两件商品，有就只看这些；它们全是用户没说过的，就是没问清需求
    clear = [cid for cid in better if bought.get(cid) == "violated"]
    cands = clear or better
    known = [cid for cid in cands if cid in in_time]
    pairs = [(cid, s) for cid in known for s in seen if s["turn"] > said[cid]["turn"]]
    # 理由优先引用当时就看得见的那条（搜索结果里显示的字段，或者商品已被打开）
    visible = [(cid, s) for cid, s in pairs if s["where"] == "被打开" or (checks.get(cid) or {}).get("evidence_field") in IN_RESULTS]
    if pairs:
        cid, after = (visible or pairs)[0]
        return {**out, "cause": "看到过正确商品没回头", "steps": [f"回合 {said[cid]['turn']}", f"S{after['step']}"],
                "explanation": f"{said_text(cid)}，正确商品满足这条，{bought_word(cid)}；之后正确商品在 S{after['step']}（第 {after['turn']} 回合）"
                               f"{after['where']}，agent 最后没有选它。"}
    if seen and clear and not known:
        return ask_more([c for c in clear if c not in in_time])

    # 用户说过正确商品更好的条件，之后正确商品却没再出现：看知道完整需求的 agent 用什么词搜到它，
    # 那些词按用户当时说过的话想得到，是搜法的问题；要用到用户没说过的条件，还是没问清需求
    known_all = [cid for cid in better if cid in in_time]
    if known_all:
        cid = known_all[0]
        if ref is None:
            ref = full_info_reference(traj.task_id, gold_asin)
            out["reference"] = ref
        if ref["queries"]:
            q = query_check(traj, ref["queries"], cutoff)
            out["facts"]["query_check"] = q
            if q["formable"]:
                return {**out, "cause": "搜索词不对", "steps": [f"回合 {said[cid]['turn']}"],
                        "explanation": f"{said_text(cid)}，正确商品满足这条，{bought_word(cid)}；但之后正确商品一直没出现在搜索结果里。"
                                       f"知道完整需求的 agent 用「{q['formable']}」搜到过它，按用户说过的话想得到这样的搜索词。"}
            return {**out, "cause": "没问清需求", "steps": [f"回合 {said[cid]['turn']}"],
                    "explanation": f"正确商品之后一直没出现在搜索结果里；知道完整需求的 agent 是用「{ref['queries'][0]}」这类词搜到它的，"
                                   f"其中「{'、'.join(q['unsaid']) or '部分条件'}」用户到最后都没说过。"}
        keys = [c for c in better if (checks.get(c) or {}).get("evidence_field") in SEARCHABLE]
        key_known = [c for c in keys if c in known_all]
        if key_known or not keys:
            cid = (key_known or known_all)[0]
            return {**out, "cause": "搜索词不对", "steps": [f"回合 {said[cid]['turn']}"],
                    "explanation": f"{said_text(cid)}，正确商品满足这条，{bought_word(cid)}；但之后正确商品一直没出现在搜索结果里。"}
        return ask_more([c for c in keys if c not in in_time], "（用户说过的条件在正确商品的标题和属性里看不出来，靠它们搜不出正确商品）")
    return ask_more([c for c in cands if c not in in_time])

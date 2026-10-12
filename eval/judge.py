"""LLM 过程评审：沿用 yyhdbl 博客的 5 个维度（0/1/2 分）和逐条 rubric 判定，评审看不到 gold 和奖励。
另外回答几个能用日志客观核对的问题，用来校准评审的准确率。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat_json  # noqa: E402
from traj import Trajectory  # noqa: E402

JUDGE_MODEL = "deepseek-v4-pro"
DIMENSIONS = ["search_strategy", "candidate_utilization", "evidence_verification", "decision_quality", "termination_efficiency"]

SYSTEM = """你是购物 agent 的轨迹评审。阅读用户需求、冻结的需求约束（rubric）和 agent 的完整执行轨迹，评价执行过程的质量，输出 JSON。
你看不到标准答案，也不需要猜测标准答案。

环境事实（评审时必须考虑）：
- 商品的标题、店铺、类目、价格区间、属性标签直接显示在搜索结果和商品详情页上。
- 商品详情页列出可选规格；选好规格后页面显示该规格的价格（当前规格价格）。
- Description 子页的内容就是商品标题，Features 和 Reviews 子页是空的，打开它们得不到任何新信息。所以“证据核验”看的是 agent 有没有对照需求核对标题、属性、规格和最终价格，而不是有没有打开子页。
- 只有出现“购买完成，任务结束”才代表真正买下了商品。agent 自己说“已购买”不算。
- 每题最多 35 个环境步骤。标注为“系统拦截，未执行”或“未执行”的调用不消耗环境步骤。

五个维度，每个打 0、1、2 分（0：存在明显问题或基本没完成；1：部分完成，仍有错误或低效；2：整体合理，没有明显问题）：
- search_strategy（搜索策略）：搜索词是否覆盖需求的关键条件，能否根据结果调整，有没有重复或无意义的搜索。
- candidate_utilization（候选利用）：是否充分利用已找到的候选，打开、比较、筛选高匹配商品，而不是发现合适候选后又从头搜索。
- evidence_verification（证据核验）：购买前是否核对了品类、品牌、型号、功能、规格和最终价格，而不是只看一眼标题。
- decision_quality（决策质量）：最终选择的商品、规格以及购买或放弃的决定是否符合需求，有没有随便购买、错误购买或无依据地放弃。
- termination_efficiency（终止与效率）：是否在合适的时机结束，有没有过早放弃、过度搜索、重复动作、无效循环，或耗尽步数。

逐条判断 rubric：satisfied、violated、unknown（轨迹中证据不足）、not_applicable。

客观问题（按轨迹事实回答，不确定就填 null）：
- purchased：agent 是否真正完成了购买
- saw_final_price_before_buy：购买前，页面上是否显示过它最终所买规格的价格（没有购买填 null）
- opened_description：agent 是否打开过任何商品的 Description 子页
- attempted_invalid_action：agent 是否做过无效动作，比如点击当前页面不存在的按钮或商品、在子页上选规格。被标为“系统拦截”的调用算无效动作；“同一回合只执行第一个商店动作”而未执行的调用不算。

每个维度和每条 rubric 判断都要在 steps 中引用相关的步骤编号（如 "S3"），没有就留空列表。

输出格式：
{"dimensions": {"search_strategy": {"score": 0, "reason": "", "steps": []}, "candidate_utilization": {...}, "evidence_verification": {...}, "decision_quality": {...}, "termination_efficiency": {...}},
 "rubric": [{"id": "c1", "status": "satisfied|violated|unknown|not_applicable", "steps": [], "reason": ""}],
 "objective": {"purchased": false, "saw_final_price_before_buy": null, "opened_description": false, "attempted_invalid_action": false},
 "summary": "一两句话总结"}"""


MULTITURN_NOTE = """

多轮设置（本题适用）：
- 用户一开始给 agent 的需求可能不完整（可能删掉了颜色、尺码、容量、数量、套餐等规格信息），用户的完整需求 agent 看不到。
- rubric 依据用户的完整需求。
{ask_rule}"""

ASK_RULE = """- agent 可以用 ask_shopper 向用户提问得知完整需求（每题最多 3 次，不消耗环境步骤），轨迹里的“[提问]”是 agent 的问题和用户的回答。
- 选错的规格如果是 agent 没问、也无法从页面推断的，要在决策质量里考虑它是不是该问而没问；问了之后有没有照着用户的回答选，也算决策质量。
- 提问本身合理就不扣分；重复问已经知道的信息、问与需求无关的内容，算作效率问题。"""

NO_ASK_RULE = """- 这一组 agent 不能向用户提问，只能根据已有信息判断。不要因为没提问扣分；评价它在信息不全时的选择是否合理（比如选最常见、最稳妥的规格，并说明依据）。"""

PAPER_NOTE = """

多轮设置（ShopSimulator 论文的多轮规则，本题适用）：
- 用户一开始只说了一个模糊的购买方向，完整需求 agent 看不到，要用 ask_shopper 提问逐步得知；用户只在被问到时才说细节，需求没问全时会拒绝购买。
- 提问和商店操作合计最多 40 个回合；轨迹里的“[提问]”是 agent 的问题和用户的回答。
- rubric 依据用户的完整需求。没问到、也无法从页面推断的需求选错了，要在决策质量里考虑是不是该问而没问；问了之后有没有照着用户的回答选，也算决策质量；用户拒绝购买后仍然下单，算决策质量问题。
- 提问本身合理就不扣分；重复问已经知道的信息、问与需求无关的内容，算作效率问题。"""


def _short(text, n) -> str:
    text = str(text or "").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


def render_state(st: dict) -> str:
    page = st.get("page_type")
    if page == "search_home":
        return "搜索首页"
    if page == "search_results":
        lines = [f"搜索结果（搜索词：{st.get('query', '')}；第 {st.get('page', 1)}/{st.get('total_pages', 1)} 页，共 {st.get('total_results', 0)} 个）"]
        for p in st.get("products") or []:
            lines.append(f"  {p.get('asin')}｜{_short(p.get('title'), 36)}｜{p.get('price')}｜属性：{_short('；'.join(p.get('key_attributes') or []), 40)}")
        return "\n".join(lines)
    if page in ("product_detail", "information_subpage"):
        p = st.get("product") or {}
        head = "商品详情页" if page == "product_detail" else f"{st.get('subpage')} 子页"
        lines = [f"{head}：{p.get('asin')}｜{p.get('title')}｜价格 {p.get('price')}｜店铺 {p.get('brand')}｜类目 {p.get('category')}",
                 f"  属性：{'；'.join(p.get('key_attributes') or [])}",
                 f"  可选规格：{json.dumps(st.get('available_options') or {}, ensure_ascii=False)}",
                 f"  已选规格：{json.dumps(st.get('selected_options') or {}, ensure_ascii=False)}"]
        if st.get("selected_price") is not None:
            lines.append(f"  当前规格价格：{st.get('selected_price')}")
        if page == "information_subpage":
            content = st.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            shown = _short(text, 120) if text and text not in ("[]", '[""]') else "（空）"
            lines.append(f"  子页内容：{shown}")
        return "\n".join(lines)
    return str(page)


def render_trajectory(traj: Trajectory) -> str:
    out = []
    for e in traj.timeline():
        if e["kind"] == "assistant":
            if e["text"]:
                out.append(f"\n回合 {e['turn']}｜agent 说：{_short(e['text'], 600)}")
            else:
                out.append(f"\n回合 {e['turn']}")
        elif e["kind"] == "step":
            result = e.get("result") or {}
            out.append(f"[S{e['step']}] 动作：{e.get('tool')} {json.dumps(e.get('args') or {}, ensure_ascii=False)}")
            out.append("购买完成，任务结束" if result.get("done") and e.get("tool") == "buy_now"
                       else ("任务结束（达到步数上限）" if result.get("done") else render_state(result.get("observation_state") or {})))
        elif e["kind"] == "ask":
            out.append(f"[提问] agent 问：{_short(e.get('question'), 300)}")
            out.append(f"       用户答：{_short(e.get('answer'), 200)}")
        elif e["kind"] == "harness" and e.get("type") in ("guard_block", "audit_block"):
            out.append(f"[系统拦截，未执行] {e.get('tool')} {json.dumps(e.get('args') or {}, ensure_ascii=False)}：{_short(e.get('reason'), 120)}")
        elif e["kind"] == "harness" and e.get("type") == "confirm_block":
            out.append(f"[系统拦截，未执行] buy_now：购买前确认，{_short('；'.join(e.get('issues') or []), 200)}")
        elif e["kind"] == "harness" and e.get("type") == "ask_hint":
            out.append(f"[系统提醒] 提示可以向用户询问规格：{'、'.join(e.get('axes') or [])}")
        elif e["kind"] == "harness" and e.get("type") == "serial_block":
            out.append(f"[未执行：同一回合只执行第一个商店动作] {e.get('tool')} {json.dumps(e.get('args') or {}, ensure_ascii=False)}")
        elif e["kind"] == "harness" and e.get("type") in ("finish_hint", "monitor_hint", "finish_nudge"):
            out.append(f"[系统提醒] {e.get('type')}")
        elif e["kind"] == "harness" and e.get("type") == "closing_soft":
            out.append(f"[系统提醒] 还剩 {e.get('left')} 个回合，请尽快下单")
        elif e["kind"] == "harness" and e.get("type") == "closing_hard":
            out.append("[系统提醒] 现在必须下单" + ("（用户已结束对话）" if e.get("bye") else f"（只剩 {e.get('left')} 个回合）"))
    if not traj.purchased:
        out.append("\n（轨迹结束，没有出现“购买完成”）")
    return "\n".join(out)


def judge(traj: Trajectory, rubric: dict) -> dict:
    cons = "\n".join(f"{c['id']} [{c['type']}/{c['hardness']}] {c['description']}" for c in rubric["constraints"])
    if traj.protocol == "paper":
        system = SYSTEM + PAPER_NOTE
        need = f"用户的开场白：{traj.agent_query}\n用户的完整需求（agent 看不到）：{traj.query}"
    elif traj.multiturn:
        system = SYSTEM + MULTITURN_NOTE.replace("{ask_rule}", ASK_RULE if traj.can_ask else NO_ASK_RULE)
        need = f"用户一开始对 agent 说的需求：{traj.agent_query}\n用户的完整需求（agent 看不到）：{traj.query}"
    else:
        system, need = SYSTEM, f"用户需求：{traj.query}"
    user = f"{need}\n\n需求约束（rubric）：\n{cons}\n\nagent 执行轨迹：\n{render_trajectory(traj)}"
    raw = chat_json(JUDGE_MODEL, system, user, max_tokens=6000, timeout=600)
    dims = raw.get("dimensions") or {}
    scores = {}
    for d in DIMENSIONS:
        try:
            scores[d] = int((dims.get(d) or {}).get("score"))
        except (TypeError, ValueError):
            scores[d] = None
    return {"scores": scores, "dimensions": dims, "rubric": raw.get("rubric") or [], "objective": raw.get("objective") or {},
            "summary": raw.get("summary", "")}

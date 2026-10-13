"""失败原因诊断：规则（process.first_error）只能指出第一次偏离成功路径的是哪一步，这里让评审模型读轨迹，说出为什么偏离：
假设只能纠正 agent 的一个行为，纠正哪一个这题就能做对，就归为哪一类原因。

评测侧能看到标准答案和完整需求，所以输入里把正确商品标出来、附上规则判定的那一步，让模型从固定的几类原因里选根本原因，
并写明依据的步骤。每道没有严格成功的题调用一次，结果按请求内容缓存。原因的分类来自对开发集失败轨迹的人工阅读。

两类原因日志就能确定，由规则直接判定、不调用模型：
- agent 选好了正确商品和规格去下单、被 harness 拦下后没能买成，判为“系统拦下了正确的购买”；
- 正确商品这次没被搜到时，拿单轮实验（agent 一开始就知道完整需求）的运行做对照，知道完整需求也从没搜到过的题判为“正确商品难搜到”。
  对照里搜到过的，把当时用的搜索词给模型，用来区分是没问到需求还是搜法不对。
"""
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import JUDGE_MODEL, render_trajectory  # noqa: E402
from llm import chat_json  # noqa: E402
from process import BUY_BLOCKS, buy_blocks  # noqa: E402
from traj import load_jsonl  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"
MIN_REFERENCE_RUNS = 3

CAUSES = {
    "没问关键需求": "用户一开始没说、agent 也没问到的需求（如预算、数量或组合装、颜色尺码），导致选了不符合的商品或规格",
    "问到了没照做": "用户在对话里明确说过的需求，agent 选的商品或规格却不符合（只看“用户在对话里说过的话”，完整需求里有、但用户没说过的不算）",
    "追查页面上没有的信息": "用户要求确认页面上根本没写的信息（详情子页是空的），agent 为此反复查看、换商品或重新搜索，耗掉了回合",
    "看到过正确商品没回头": "正确商品已经出现在搜索结果里或被打开过，agent 后来没有回头比较或选择它",
    "搜索跑偏": "搜索词没抓住关键条件，或者塞了太多条件，正确商品一直没搜到",
    "规格没点": "下单时有规格项没有点选（常见于只有一个选项的规格）",
    "规格选错": "规格项点选了，但选的不是用户要的那个",
    "回合用完被迫下单": "回合或环境步数快用完，被要求立即下单，只好买了不对的商品；如果回合是被前面某个原因耗掉的，选那个原因",
    "动作无效": "想打开正确商品或选正确规格，但动作在当前页面无效，被环境忽略",
    "替代品也满足需求": "买到的商品也满足用户的全部需求，只是不是标准答案那一件",
    "标准答案有问题": "标准答案本身和用户需求矛盾",
    "其他": "以上都不是，在 explanation 里说明",
    "正确商品难搜到": "知道完整需求的 agent（单轮实验的多次运行）也从没搜到过正确商品：难在检索，不是这次哪一步做错了（由规则判定，不交给模型选）",
    "系统拦下了正确的购买": "agent 已经选好正确商品和全对的规格去下单，被 harness 的检查拦下，之后没能买成（由规则判定，不交给模型选）",
}
RULE_ONLY = {"正确商品难搜到", "系统拦下了正确的购买"}

SYSTEM = """你是购物 agent 的失败分析员。这道题 agent 没有严格成功：没买到标准答案那件商品，或者规格没有全对。
你能看到用户的完整需求和拆好的约束、标准答案、agent 实际买了什么、评测结果、规则判定的“第一次偏离成功路径”的那一步，以及完整轨迹。轨迹里正确商品的 ID 后面标了【正确商品】。

环境事实：
- 商品的标题、店铺、类目、价格、属性直接显示在搜索结果和详情页上；Description 子页的内容就是标题，Features 和 Reviews 子页是空的。
- 打开商品只能在包含它的那页搜索结果上进行；不在当前页就要重新搜到。
- 多轮设置里，用户一开始只说模糊方向，只有被问到才说细节；需求没问全时用户会拒绝购买。提问和操作合计最多 40 个回合，环境步数最多 35 步。
- 系统在快没回合时会要求立即下单。

区分清楚：“完整需求”只有用户自己知道，agent 看不到；agent 只知道“用户在对话里说过的话”。用户没说过的需求没被满足，是“没问关键需求”，不是“问到了没照做”。

请这样判断：假设只能纠正 agent 的一个行为，纠正哪一个，这道题最有可能就做对了？选这个行为对应的原因。几个都可以时，选最早发生的那个（比如去追查页面上没有的信息、结果回合用完被迫下单，选前者）。
纠正必须是 agent 当时就做得到的：只能用到那时用户已经说过的话（看回合号）和页面上写的内容；页面上写着、但用户还没提过的条件，不算用户的需求，不能用“这件就是标准答案”这一点（agent 看不到标准答案）。所以：
- 按用户当时已经说过的需求，正确商品明显比 agent 选的更合适（比如页面上写着用户要的条件），纠正是“回头选它”，归为“看到过正确商品没回头”；
- 按用户当时说过的需求，正确商品和 agent 选的那件看不出谁更好，纠正应该是“先问清那条需求”，归为“没问关键需求”。
正确商品这次没被搜到时，输入里会给出对照：一开始就知道完整需求的 agent 用什么词搜到过它。搜到它的词如果用到了用户这次还没说过的条件，是“没问关键需求”；如果只用到用户已经说过的条件，是“搜索跑偏”。
原因只能从下面选一个：
{causes}

输出 JSON：{{"cause": "原因名称", "secondary": ["其他也有份的原因名称"], "steps": ["S3", "回合 12"], "explanation": "一两句话：应该在哪一步怎么做，这题就能做对"}}"""


# 日志能确定的事实先由规则排除不可能的原因，模型只在剩下的里面选
ONLY_IF_BOUGHT_GOLD = {"规格没点", "规格选错"}
NOT_IF_BOUGHT_GOLD = {"看到过正确商品没回头", "搜索跑偏", "替代品也满足需求"}


def allowed_causes(rec: dict) -> list:
    if rec.get("bought_gold"):
        return [c for c in CAUSES if c not in NOT_IF_BOUGHT_GOLD | RULE_ONLY]
    banned = ONLY_IF_BOUGHT_GOLD | RULE_ONLY
    if rec.get("retrieved_at") is None:
        banned = banned | {"看到过正确商品没回头"}
    return [c for c in CAUSES if c not in banned]


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


def _gold_seen(traj, rec: dict) -> str:
    def at(i):  # retrieved_at / opened_at 从 1 开始数
        s = traj.steps[i - 1]
        return f"S{s.get('step')}（第 {s.get('turn')} 回合）"
    seen = f"第一次出现在搜索结果里是 {at(rec['retrieved_at'])}" if rec.get("retrieved_at") is not None else "从没出现在任何一页搜索结果里"
    opened = f"第一次被打开是 {at(rec['opened_at'])}" if rec.get("opened_at") is not None else "从没被打开"
    return f"正确商品{seen}；{opened}"


def _system(causes: list) -> str:
    return SYSTEM.format(causes="\n".join(f"- {k}：{CAUSES[k]}" for k in causes))


def _product_line(asin: str, products: dict) -> str:
    p = products.get(str(asin)) or {}
    return f"[{asin}] {p.get('title', '（商品库里没有）')}｜店铺 {p.get('shop', '')}｜价格 {p.get('price_range', '')}"


def diagnose(traj, rubric: dict, gold: dict, rec: dict, products: dict) -> dict:
    gold_asin = str(gold["asin"])
    stopped = next((b for b in buy_blocks(traj, gold) if b["was_correct"]), None)
    if stopped:
        return {"cause": "系统拦下了正确的购买", "secondary": [], "steps": [f"回合 {stopped['turn']}"],
                "explanation": f"第 {stopped['turn']} 回合 agent 已经选好正确商品和规格去下单，被系统的{BUY_BLOCKS[stopped['type']]}拦下，之后没能买成。"}
    ref = full_info_reference(traj.task_id, gold_asin) if rec.get("retrieved_at") is None else None
    if ref and ref["runs"] >= MIN_REFERENCE_RUNS and not ref["retrieved"]:
        return {"cause": "正确商品难搜到", "secondary": [], "steps": [], "reference": ref,
                "explanation": f"一开始就知道完整需求的 agent（单轮实验 {ref['runs']} 次运行）也从没搜到过正确商品，这题难在检索。"}
    cons = "\n".join(f"{c['id']} [{c['type']}/{c['hardness']}] {c['description']}" for c in rubric["constraints"])
    if traj.purchased:
        p = traj.purchase
        bought = f"{_product_line(p.get('asin'), products)}｜所选规格 {json.dumps(p.get('options') or {}, ensure_ascii=False)}｜成交价 {p.get('price')}"
    else:
        bought = "没有买"
    unmet = [f"{r['id']}（{r.get('verdict')}）" for r in rec.get("check") or [] if r.get("verdict") != "satisfied"]
    verdict = "买到的商品可以接受" if rec.get("acceptable_core") else "买到的商品不可接受"
    fe = rec.get("first_error") or {}
    first = (f"{fe.get('type')}，在 S{fe.get('step')}" if fe.get("step") else str(fe.get("type"))) if fe else "无"
    need = f"用户的开场白：{traj.agent_query}\n用户的完整需求（agent 看不到）：{traj.query}" if traj.agent_query != traj.query else f"用户需求：{traj.query}"
    said = "\n".join([f"- 开场：{traj.agent_query}"] + [f"- 回合 {q.get('turn')}：{q['answer']}" for q in traj.asks])
    step_turn = next((s.get("turn") for s in traj.steps if s.get("step") == fe.get("step")), None) if fe else None
    user = (
        f"{need}\n\n用户在对话里说过的话（按顺序）：\n{said}\n\n需求约束：\n{cons}\n\n"
        f"标准答案：{_product_line(gold_asin, products)}｜正确规格 {json.dumps(gold.get('options') or [], ensure_ascii=False)}\n"
        f"agent 买的：{bought}\n"
        f"评测结果：{verdict}；没有满足或无法确认的约束：{'、'.join(unmet) or '无'}\n"
        f"规则判定的第一次偏离：{first}" + (f"（第 {step_turn} 回合；那时用户已经说过的话，是上面标了更早回合的那些）" if step_turn else "") + "\n"
        f"{_gold_seen(traj, rec)}\n"
        + (f"对照：一开始就知道完整需求的 agent（单轮实验 {ref['runs']} 次运行）里，{ref['retrieved']} 次搜到过正确商品，"
           f"搜到时用的词：{'、'.join(f'「{q}」' for q in ref['queries'])}\n" if ref and ref["retrieved"] else "")
        + "\n"
        f"agent 执行轨迹：\n{render_trajectory(traj).replace(gold_asin, f'{gold_asin}【正确商品】')}"
    )
    causes = allowed_causes(rec)
    raw = chat_json(JUDGE_MODEL, _system(causes), user, max_tokens=1500, timeout=600)
    cause = raw.get("cause") if raw.get("cause") in causes else "其他"
    out = {
        "cause": cause,
        "secondary": [c for c in raw.get("secondary") or [] if c in causes and c != cause],
        "steps": [str(s) for s in raw.get("steps") or []],
        "explanation": str(raw.get("explanation") or ""),
    }
    if ref:
        out["reference"] = ref
    return out

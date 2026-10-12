#!/usr/bin/env python3
"""打印一条轨迹：需求、每回合的模型输出与工具调用、返回页面摘要、harness 事件、购买结果与 gold 对比。

用法：python3 scripts/show_traj.py runs/<run>/task_<id> [--full]
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data" / "private" / "gold.jsonl"


def norm(text) -> str:
    # 环境渲染规格值时会把 "/" 显示成 " | "，比较前统一
    return re.sub(r"\s+", "", str(text)).lower().replace("|", "/")


def load_gold() -> dict:
    if not GOLD_PATH.exists():
        return {}
    rows = (json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
    return {row["task_id"]: row for row in rows}


def strict_success(purchase: dict, gold: dict) -> bool:
    if not purchase or str(purchase.get("asin")) != str(gold["asin"]):
        return False
    selected = [norm(v) for v in (purchase.get("options") or {}).values()]

    def matches(gold_value) -> bool:
        # 少数 gold 标注里的 emoji 被编码成了 "?"，当作单字符通配
        pattern = "".join("." if ch == "?" else re.escape(ch) for ch in norm(gold_value))
        return any(re.fullmatch(pattern, v, re.S) for v in selected)

    return all(matches(o) for o in gold["options"])


def session_messages(task_dir: Path) -> list:
    files = sorted((task_dir / "session").glob("*.jsonl"))
    if not files:
        return []
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r["message"] for r in rows if r.get("type") == "message"]


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if c.get("type") == "text")


def summarize_page(text: str, full: bool) -> str:
    lines = text.splitlines()
    if full or len(lines) <= 14:
        return text
    head = lines[:12]
    return "\n".join(head + [f"...（省略 {len(lines) - 13} 行）", lines[-1]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--eval", action="store_true", help="同时打印评测结果（需先运行 eval/evaluate.py）")
    args = parser.parse_args()

    task_dir = args.task_dir
    task_id = int(task_dir.name.split("_")[-1])
    gold = load_gold().get(task_id)
    messages = session_messages(task_dir)

    reset_file = task_dir / "reset_public.json"
    reset = json.loads(reset_file.read_text(encoding="utf-8")) if reset_file.exists() else {}
    multiturn = reset.get("agent_query") not in (None, reset.get("query"))
    tokens_in = tokens_cached = tokens_out = turns = 0
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            label = "需求" if turns == 0 else "用户消息"
            print(f"【{label}】{text_of(msg.get('content'))}\n")
            if multiturn and turns == 0:
                print(f"【完整需求（agent 看不到，只有模拟买家知道）】{reset.get('query')}\n")
        elif role == "assistant":
            turns += 1
            usage = msg.get("usage") or {}
            tokens_in += usage.get("input", 0)
            tokens_cached += usage.get("cacheRead", 0)
            tokens_out += usage.get("output", 0)
            said = text_of(msg.get("content")).strip()
            calls = [c for c in msg.get("content") or [] if c.get("type") == "toolCall"]
            prompt_tokens = usage.get("input", 0) + usage.get("cacheRead", 0)
            print(f"── 回合 {turns}  (prompt={prompt_tokens} out={usage.get('output', 0)} stop={msg.get('stopReason')})")
            if said:
                print(f"  模型：{said if args.full else said[:300]}")
            for c in calls:
                print(f"  调用：{c.get('name')} {json.dumps(c.get('arguments'), ensure_ascii=False)}")
        elif role == "toolResult":
            flag = " [错误]" if msg.get("isError") else ""
            body = summarize_page(text_of(msg.get("content")), args.full)
            print(f"  返回{flag}（{msg.get('toolName')}）：")
            print("    " + body.replace("\n", "\n    "))

    events_file = task_dir / "harness_events.jsonl"
    if events_file.exists() and events_file.read_text().strip():
        print("\n【harness 事件】")
        for line in events_file.read_text().splitlines():
            print("  " + line)

    print(f"\n【统计】LLM 回合 {turns}，prompt token {tokens_in + tokens_cached}（其中缓存命中 {tokens_cached}），输出 token {tokens_out}")
    terminal_file = task_dir / "terminal.json"
    if not terminal_file.exists():
        print("【结果】没有到达结束页（未购买）")
    else:
        result = json.loads(terminal_file.read_text())["result"]
        purchase = result.get("purchase") or {}
        print(f"【结果】结束原因：{result.get('termination_reason')}，环境步数：{json.loads(terminal_file.read_text())['step']}")
        print(f"  买了：{purchase.get('asin')} {purchase.get('name', '')} 规格={purchase.get('options')} 价格={purchase.get('price')}")
        if gold:
            print(f"  gold：{gold['asin']} {gold['title']} 规格={gold['options']}")
            print(f"  严格成功：{strict_success(purchase, gold)}")
    if args.eval:
        print_eval(task_dir)


def print_eval(task_dir: Path) -> None:
    eval_file = task_dir.parent / "eval" / "tasks.jsonl"
    task_id = int(task_dir.name.split("_")[-1])
    rec = None
    if eval_file.exists():
        for line in eval_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["task_id"] == task_id:
                rec = row
    if rec is None:
        print("\n（没有评测结果，先运行 eval/evaluate.py）")
        return
    rubric_path = ROOT / "data" / "rubrics" / f"{task_id}.json"
    rubric = json.loads(rubric_path.read_text(encoding="utf-8")) if rubric_path.exists() else {"constraints": []}
    check = {c["id"]: c for c in rec.get("check") or []}
    ver = rec.get("verification") or {}
    print("\n【评测结果】")
    print(f"  严格成功 {rec['strict_success']}｜可接受（core）{rec.get('acceptable_core')}｜成功分档 {rec.get('success_class')}｜标注矛盾题 {rec.get('gold_conflict')}")
    print("  需求约束与核对（购买前核验状态）：")
    for c in rubric["constraints"]:
        r = check.get(c["id"], {})
        print(f"    {c['id']} [{c['type']}/{c['hardness']}] {c['description']}  → {r.get('verdict', '-')}"
              f"（{ver.get(c['id'], '-')}）  依据 {r.get('evidence_field', '')}: {str(r.get('evidence_quote', ''))[:30]}")
    fe = rec.get("first_error")
    if fe:
        steps = [json.loads(l) for l in (task_dir / "env_steps.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        at = next((s for s in steps if s.get("step") == fe.get("step")), None)
        where = f"第 {fe.get('step')} 步 {at['tool']} {json.dumps(at.get('args'), ensure_ascii=False)}" if at else "（整条轨迹）"
        print(f"  首错：{fe['type']}，{where}")
    cl = rec.get("claims") or {}
    print(f"  自述：声称已买 {cl.get('claims_purchased')}｜假装已买 {cl.get('false_purchase_claim')}｜虚假核验 {cl.get('false_verification')}｜价格不符 {cl.get('price_mismatch')}")
    a = rec.get("asking")
    if a:
        what = "开场没说的约束" if a["group"] == "paper" else "被删的约束"
        print(f"  提问：{a['group']} 组，{what} {a['hidden']}，问到 {a['hidden_asked']}，购买时没满足 {a['hidden_unmet']}，失败归因 {a['ask_failure']}")
        for q in a["questions"]:
            chk = q["answer_check"]
            judged = f"｜买家回答：针对 {chk.get('relevant')} 忠实 {chk.get('faithful')} 泄露 {chk.get('leaked')}" if chk else ""
            print(f"    [{q['kind']}] 问到 {q['targets']}｜问：{q['question'][:80]}｜答：{q['answer']}{judged}")
    j = rec.get("judge")
    if j:
        print(f"  LLM 评审：{j['scores']}")
        print(f"    {j.get('summary', '')}")
        print(f"    评审客观题与日志一致：{rec.get('judge_calibration')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""论文多轮设置的开场白：照官方 multi_eval，先对模拟买家说"请提供您的购物需求。"，买家的回复就是 agent 收到的第一句话。

每题只生成一次并冻结，所有对比组用同一句开场白，保证配对比较。
完整需求和标准答案的规格选项从环境 reset 拿（官方也是这样喂给买家的），规格选项只进买家的提示，不写进输出文件。

用法：python3 scripts/make_paper_openings.py --tasks data/dev_ids.json --out data/multiturn_paper/dev_openings.jsonl
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "runner"))
from llm import chat_text  # noqa: E402
from paper_protocol import OPENING_REQUEST, shopper_system  # noqa: E402
from run import env_call, query_from_reset  # noqa: E402


def reset_with_retry(env_url: str, task_id: int) -> dict:
    for attempt in range(30):
        try:
            return env_call(env_url, {"action": "reset", "idx": task_id})
        except RuntimeError as exc:
            if "available environment resource" not in str(exc) or attempt == 29:
                raise
            time.sleep(10)
    raise RuntimeError("unreachable")


def build(task_id: int, args) -> dict:
    reset = reset_with_retry(args.env_url, task_id)
    try:
        query = query_from_reset(reset)
        system = shopper_system(query, reset.get("goal_options"))
    finally:
        env_call(args.env_url, {"action": "release_one", "env_idx": reset["env_idx"]})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": OPENING_REQUEST}]
    opening = chat_text(args.shopper_model, messages, max_tokens=args.max_tokens)
    if not opening:
        raise RuntimeError(f"task {task_id}: 开场白为空")
    return {"task_id": task_id, "full_query": query, "opening": opening, "shopper_model": args.shopper_model}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="题号列表（json）")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shopper-model", default="qwen3.7-plus")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    task_ids = json.loads(Path(args.tasks).read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda t: build(t, args), task_ids))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lengths = sorted(len(r["opening"]) for r in records)
    print(f"{len(records)} 题，开场白长度中位数 {lengths[len(lengths) // 2]} 字（最短 {lengths[0]}，最长 {lengths[-1]}）；写入 {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""导出题目 id 和离线答案表。

- data/test_ids.json：作者的 Final-200 Clean（data/evaluation/tasks.jsonl）
- data/dev_ids.json：从全量商品库中随机抽取，与测试集在 id 和需求文本上都不重叠
- data/persona_test_ids.json、data/persona_dev_ids.json：画像设置的题，从官方画像测试集（1,343 道）中随机抽取，
  与上面两个集合在 id 和需求文本上都不重叠
- data/private/gold.jsonl：只给评测器读，agent 侧永远不读这个目录
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTHOR = ROOT / "third_party" / "shopping-grpo-longhorizon"
DEFAULT_CATALOG = AUTHOR / "environments/ShopSimulator/shop_env/data/items_eval_train.json"
DEFAULT_TEST = AUTHOR / "data/evaluation/tasks.jsonl"
DEFAULT_PERSONA_EVAL = ROOT / "third_party" / "shopsim_hf" / "fine_items_eval_persona.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--test-tasks", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--n-dev", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--persona-eval", type=Path, default=DEFAULT_PERSONA_EVAL)
    parser.add_argument("--n-persona-test", type=int, default=200)
    parser.add_argument("--n-persona-dev", type=int, default=50)
    parser.add_argument("--persona-seed", type=int, default=20261014)
    args = parser.parse_args()

    items = json.loads(args.catalog.read_text(encoding="utf-8"))
    test_ids = [json.loads(line)["task_id"] for line in args.test_tasks.read_text().splitlines() if line.strip()]
    if len(set(test_ids)) != len(test_ids):
        raise SystemExit("测试集 id 有重复")

    test_texts = {items[i]["instructions"][0]["instruction"] for i in test_ids}
    candidates = [
        i for i in range(len(items))
        if i not in set(test_ids) and items[i]["instructions"][0]["instruction"] not in test_texts
    ]
    dev_ids = sorted(random.Random(args.seed).sample(candidates, args.n_dev))

    used = set(test_ids) | set(dev_ids)
    used_texts = {items[i]["instructions"][0]["instruction"] for i in used}
    persona_asins = {str(json.loads(line)["asin"]) for line in args.persona_eval.read_text(encoding="utf-8").splitlines() if line.strip()}
    persona_candidates = [
        i for i in range(len(items))
        if str(items[i]["asin"]) in persona_asins and i not in used and items[i]["instructions"][0]["instruction"] not in used_texts
    ]
    picked = random.Random(args.persona_seed).sample(persona_candidates, args.n_persona_test + args.n_persona_dev)
    persona_test_ids = sorted(picked[: args.n_persona_test])
    persona_dev_ids = sorted(picked[args.n_persona_test:])

    data_dir = ROOT / "data"
    private_dir = data_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "test_ids.json").write_text(json.dumps(test_ids) + "\n")
    (data_dir / "dev_ids.json").write_text(json.dumps(dev_ids) + "\n")
    (data_dir / "persona_test_ids.json").write_text(json.dumps(persona_test_ids) + "\n")
    (data_dir / "persona_dev_ids.json").write_text(json.dumps(persona_dev_ids) + "\n")

    with (private_dir / "gold.jsonl").open("w", encoding="utf-8") as f:
        for split, ids in (("test", test_ids), ("dev", dev_ids), ("persona_test", persona_test_ids), ("persona_dev", persona_dev_ids)):
            for i in ids:
                item = items[i]
                ins = item["instructions"][0]
                row = {
                    "task_id": i,
                    "split": split,
                    "asin": item["asin"],
                    "options": ins["instruction_options"],
                    "instruction": ins["instruction"],
                    "attributes": ins.get("attributes", []),
                    "title": item["title"],
                    "category": item["category"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"test={len(test_ids)} dev={len(dev_ids)} candidates={len(candidates)} seed={args.seed}")
    print(f"persona_test={len(persona_test_ids)} persona_dev={len(persona_dev_ids)} candidates={len(persona_candidates)} seed={args.persona_seed}")


if __name__ == "__main__":
    main()

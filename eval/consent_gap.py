"""问完用户之后又在商店里做了几个动作才下单，和下单时用户有没有点头的关系。

只读评测结果（runs/<run>/eval/tasks.jsonl 里的 consent，见 eval/consent.py）和轨迹，不调模型。
动作数 = 下单回合 − 最后一次问用户的回合 − 1（被合法性校验拦下的动作也算）。
收尾单（下单前出现过"现在必须下单"提示，按官方规则不需要用户同意）单独统计。

    python3 eval/consent_gap.py runs/test_paper_v2 runs/test_paper_off
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj import Trajectory, load_jsonl  # noqa: E402

BUCKETS = ("0", "1-3", "4-8", "9+")
CLOSING_EVENTS = {"closing_hard", "roles_close_hard"}


def bucket(gap: int) -> str:
    actions = gap - 1
    return "0" if actions <= 0 else "1-3" if actions <= 3 else "4-8" if actions <= 8 else "9+"


def main(run_dirs: list) -> None:
    for run in map(Path, run_dirs):
        groups = {"对话中途": defaultdict(Counter), "收尾": defaultdict(Counter)}
        for rec in load_jsonl(run / "eval" / "tasks.jsonl"):
            c = rec.get("consent")
            if not c or c.get("turns_before_buy") is None:
                continue
            task_dir = run / f"task_{rec['task_id']}"
            buy_turn = Trajectory(task_dir).buy_turn()
            closing = buy_turn is not None and any(
                e.get("type") in CLOSING_EVENTS and int(e.get("turn") or 0) <= buy_turn
                for e in load_jsonl(task_dir / "harness_events.jsonl")
            )
            groups["收尾" if closing else "对话中途"][bucket(c["turns_before_buy"])][c["attitude"] == "agree"] += 1
        print(run.name)
        for name, by in groups.items():
            cells = []
            for b in BUCKETS:
                n = by[b][True] + by[b][False]
                cells.append(f"{b} 个动作：{by[b][False]}/{n}" + (f"（{by[b][False] / n:.0%}）" if n else ""))
            print(f"  {name}的下单，用户没点头：" + "；".join(cells))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])

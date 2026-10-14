"""改了评测代码以后，核对已评过的运行的评审输入有没有变：用当前代码重建每题的评审请求，只查缓存、不调模型。
全部命中 = 输入和当初评审时一字不差，评审分数也就不会变。

用法：python3 scripts/check_judge_cache.py runs/test_paper_v2 [runs/<run> ...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import llm  # noqa: E402
from evaluate import load_rubric  # noqa: E402
from judge import judge  # noqa: E402
from traj import Trajectory  # noqa: E402


class Miss(Exception):
    pass


def no_call(payload, timeout):
    raise Miss()


def main() -> None:
    llm._post = no_call
    changed = False
    for run in sys.argv[1:]:
        hit, miss = 0, []
        for d in sorted(Path(run).glob("task_*")):
            traj = Trajectory(d)
            try:
                judge(traj, load_rubric(traj.task_id))
                hit += 1
            except Miss:
                miss.append(traj.task_id)
        changed = changed or bool(miss)
        print(f"{Path(run).name}: 命中缓存 {hit}，没命中 {len(miss)}" + (f"（{miss[:10]}）" if miss else ""))
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()

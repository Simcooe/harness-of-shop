#!/usr/bin/env bash
# Final-200 测试集重复实验（第二遍用）：原版 / 插件 v1 / 只开 guard，各重复 3 次（共 1800 次任务运行）。
# 组件选择已在开发集上完成；测试集只跑这一次，不再据此调整插件。
set -euo pipefail
cd "$(dirname "$0")/.."

for rep in 1 2 3; do
  for cfg in off all guard; do
    name="test_${cfg}_r${rep}"
    python3 runner/run.py --run-name "$name" --tasks data/test_ids.json --workers 16 --plugin "$cfg" --timeout 900 | tail -1
    python3 eval/evaluate.py "runs/$name" > /dev/null
  done
done

for rep in 1 2 3; do
  python3 eval/compare.py "runs/test_off_r${rep}" "runs/test_all_r${rep}" | sed -n 1,3p
  python3 eval/compare.py "runs/test_off_r${rep}" "runs/test_guard_r${rep}" | sed -n 1,3p
done

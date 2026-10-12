#!/usr/bin/env bash
# 开发集上的噪声估计 + 逐个去掉组件的消融。用法：bash scripts/run_ablation.sh [tasks_file]
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS="${1:-data/dev_ids.json}"
ALL=(guard ledger monitor audit finish contract)

run() {
  python3 runner/run.py --run-name "$1" --tasks "$TASKS" --workers 16 --plugin "$2" --plugin-file pi-shop/plugin_v0.ts --timeout 900 | tail -1
  python3 eval/evaluate.py "runs/$1" > /dev/null
}

run dev100_off_r2 off
run dev100_on_r2 all
for drop in "${ALL[@]}"; do
  keep=$(printf '%s\n' "${ALL[@]}" | grep -vx "$drop" | paste -sd, -)
  run "dev100_no_${drop}" "$keep"
done
echo "ablation done"
